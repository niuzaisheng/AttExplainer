from typing import List
import numpy as np
import random
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from accelerate.utils import send_to_device
import torch.optim as optim
from per.prioritized_memory import Memory
from torch.nn.utils.rnn import pad_sequence


class DQNNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.bins_num = config.bins_num
        self.matrix_num = 1

        self.h1 = self.matrix_num * (self.bins_num * 2 + 4)
        self.h2 = 63
        self.h3 = 8
        self.layer1 = nn.Linear(self.h1, self.h2)
        self.layer2 = nn.Linear(self.h2 + 1, self.h3)
        self.out = nn.Linear(self.h3, 1)

    def forward(self, x, s, seq_len):
        batch_size, matrix_num, max_seq_len, _ = x.size()

        x = x.reshape(batch_size, matrix_num, max_seq_len, self.bins_num * 2 + 4)  # [B, matrix_num, seq, (bins_num * 2 + 4)]
        x = x.permute(0, 2, 3, 1)  # [Batch x seq x (bins_num+2) x matrix_num]
        x = x.reshape(batch_size, max_seq_len, self.h1)  # [B, matrix_num, seq, 8]
        x = self.layer1(x)  # [B, seq, bins_num x 4]
        x = torch.relu(x)
        x = F.dropout(x)
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, h2 + 1]
        x = self.layer2(x)
        x = torch.relu(x)
        x = F.dropout(x)
        x = self.out(x)
        return x.reshape(batch_size, max_seq_len)  # [B , seq]


def gather2D(tensors: List[Tensor]):
    out_dict = {}
    first = tensors[0]
    batch_size = len(tensors)
    for k in first.keys():
        if "attentions" in k or "observation" in k:
            head_num = first[k].size(0)
            max_seq_len = max([item[k].size(1) for item in tensors])  # [12 x seq x seq]
            attentions = []
            for i, item in enumerate(tensors):
                item_length = item[k].size(1)
                temp = F.pad(item[k], (0, 0, 0, max_seq_len-item_length))
                attentions.append(temp)
            attentions = torch.stack(attentions)
            out_dict[k] = attentions
        else:
            if "done" in k or k in ["actions", "rewards"]:
                out_dict[k] = torch.stack([item[k] for item in tensors])
            elif k != "seq_length":
                out_dict[k] = pad_sequence([item[k] for item in tensors], batch_first=True)
    return out_dict


BufferItem = namedtuple("BufferItem", ("all_attentions", "next_attentions", "actions", "game_status", "next_game_status",  "seq_length", "rewards", "ifdone"))


class DQN(object):
    """
        DQN progress for training
    """

    def __init__(self, config, mask_token_id=103):

        self.device = torch.device("cuda", config.gpu_index)
        # self.device = torch.device("cpu")
        self.mask_token_id = mask_token_id

        self.eval_net = DQNNet(config)
        self.target_net = DQNNet(config)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=config.dqn_rl)

        self.eval_net = send_to_device(self.eval_net, self.device)
        self.target_net = send_to_device(self.target_net, self.device)

        self.learn_step_counter = 0
        self.max_memory_capacity = config.max_memory_capacity
        self.target_replace_iter = config.target_replace_iter
        self.dqn_batch_size = config.dqn_batch_size
        self.epsilon = config.epsilon
        self.gamma = config.gamma
        self.use_categorical_policy = config.use_categorical_policy
        self.memory = Memory(self.max_memory_capacity)
        self.loss_func = nn.MSELoss(reduce=False)

    def choose_action(self, batch, batch_seq_length, special_tokens_mask, all_attentions, game_status, no_random=False):
        target_device = batch["input_ids"].device
        self.eval_net.eval()
        with torch.no_grad():
            actions = self.eval_net(all_attentions, game_status, batch_seq_length).detach()
            for i, index in enumerate(batch_seq_length):
                if no_random is not True and np.random.uniform() > self.epsilon:
                    actions[i] = 0
                    actions[i][random.randint(0, index-1)] = 1

        actions = actions.masked_fill(special_tokens_mask, -np.inf)

        if self.use_categorical_policy:
            # categorical policy
            actions = torch.softmax(actions, dim=1)
            m = torch.distributions.categorical.Categorical(probs=actions)
            select_action = m.sample()
        else:
            # argmax policy
            select_action = torch.argmax(actions, dim=1)  # [B, seq_len]

        next_game_status = game_status.clone()
        for i, position in enumerate(select_action):
            if game_status[i, position] == 0:
                next_game_status[i, position] = 1
            else:
                next_game_status[i, position] = 0

        mask_map = next_game_status == 1

        # On target device:
        mask_map = mask_map.to(target_device)
        post_input_ids = batch["input_ids"].clone()
        post_input_ids = post_input_ids.masked_fill(~mask_map, self.mask_token_id)

        for i, index in enumerate(batch_seq_length):
            post_input_ids[i, index:] = 0

        post_batch = {"input_ids": post_input_ids, "attention_mask": batch["attention_mask"], "token_type_ids": batch["token_type_ids"]}

        return post_batch, select_action, next_game_status

    def initial_action(self, batch, special_tokens_mask, seq_length, batch_max_seq_length, device):
        batch_size = len(seq_length)
        actions = torch.zeros((batch_size, batch_max_seq_length))
        game_status = torch.ones((batch_size, batch_max_seq_length))
        for i, index in enumerate(seq_length):
            game_status[i, index:] = 0
        return batch, actions.to(device), game_status.to(device)

    def store_transition(self, batch_size, all_attentions, next_attentions, game_status, next_game_status, actions, batch_seq_length, rewards, ifdone):
        self.eval_net.eval()
        with torch.no_grad():
            q_loss = self.get_td_loss(batch_seq_length, all_attentions, next_attentions, game_status, next_game_status, actions, rewards, ifdone).detach()

        all_attentions = send_to_device(all_attentions, "cpu")
        next_attentions = send_to_device(next_attentions, "cpu")
        game_status = send_to_device(game_status, "cpu")
        next_game_status = send_to_device(next_game_status, "cpu")
        actions = send_to_device(actions, "cpu")
        rewards = send_to_device(rewards, "cpu")
        ifdone = send_to_device(ifdone, "cpu")
        q_loss = send_to_device(q_loss, "cpu").numpy()

        temp_list = []
        for i in range(batch_size):
            new_item = BufferItem(all_attentions=all_attentions[i],
                                  next_attentions=next_attentions[i],
                                  actions=actions[i],
                                  game_status=game_status[i],
                                  next_game_status=next_game_status[i],
                                  seq_length=batch_seq_length[i],
                                  rewards=rewards[i],
                                  ifdone=ifdone[i],
                                  )
            temp_list.append(new_item)
            self.memory.add(q_loss[i], temp_list[i])

    def get_td_loss(self, batch_seq_length, all_attentions, next_attentions, game_status, next_game_status, actions, rewards, ifdone):
        ifdone = ifdone.float()
        q_eval = self.eval_net(all_attentions, game_status, batch_seq_length)
        max_seq_len = q_eval.size(1)
        actions = F.one_hot(actions, num_classes=max_seq_len).bool()

        q_eval = q_eval.masked_select(actions)  # [B, seq, 1]  -> [B, 1]

        with torch.no_grad():
            q_target = self.target_net(next_attentions, next_game_status, batch_seq_length)
        q_target = q_target.masked_select(actions)

        q_target = rewards + self.gamma * (1 - ifdone) * q_target.detach()

        q_loss = self.loss_func(q_eval, q_target)   # [B, seq]

        return q_loss  # [B] [B, seq] [B, seq]

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        samples, sample_idxs, is_weight = self.memory.sample(self.dqn_batch_size)
        is_weight = torch.FloatTensor(is_weight).reshape(self.dqn_batch_size, 1).to(self.device)

        samples = [item._asdict() for item in samples]
        batch_seq_length = [item["seq_length"] for item in samples]
        samples = gather2D(samples)
        samples = send_to_device(samples, self.device)
        all_attentions = samples["all_attentions"]
        next_attentions = samples["next_attentions"]
        game_status = samples["game_status"]
        next_game_status = samples["next_game_status"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        ifdone = samples["ifdone"]

        self.eval_net.train()
        q_loss = self.get_td_loss(batch_seq_length, all_attentions, next_attentions, game_status, next_game_status, actions, rewards, ifdone)

        q_loss_detached = q_loss.detach()
        for i in range(self.dqn_batch_size):
            idx = sample_idxs[i]
            self.memory.update(idx, q_loss_detached[i].item())

        dqn_loss = q_loss * is_weight
        dqn_loss = dqn_loss.mean()
        dqn_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return dqn_loss.detach().item()


class DQN_eval(object):
    """
        DQN progress for eval
    """

    def __init__(self, config, mask_token_id=103):

        self.device = torch.device("cpu")
        self.eval_net = DQNNet(config)
        self.mask_token_id = mask_token_id
        with open(config.dqn_weights_path, "rb") as f:
            self.eval_net.load_state_dict(torch.load(f))

    def choose_action_for_eval(self, batch, batch_seq_length, special_tokens_mask, all_attentions, game_status):
        target_device = batch["input_ids"].device
        batch_size, _, seq_len, _ = all_attentions.size()
        self.eval_net.eval()
        all_attentions = send_to_device(all_attentions, self.device)
        with torch.no_grad():
            actions = self.eval_net(all_attentions, game_status, batch_seq_length).detach()
        actions = actions.masked_fill(special_tokens_mask, -np.inf)
        select_action = torch.argmax(actions, dim=1)  # [B, seq_len]

        next_game_status = game_status.clone()
        for i, position in enumerate(select_action):
            if game_status[i, position].item() == 0:
                next_game_status[i, position] = 1
            else:
                next_game_status[i, position] = 0

        mask_map = next_game_status == 1

        # On target device:
        mask_map = mask_map.to(target_device)
        post_input_ids = batch["input_ids"].clone().detach()
        post_input_ids = post_input_ids.masked_fill(~mask_map, self.mask_token_id)

        for i, index in enumerate(batch_seq_length):
            post_input_ids[i, index:] = 0

        post_batch = {"input_ids": post_input_ids,
                      "attention_mask": batch["attention_mask"],
                      "token_type_ids": batch["token_type_ids"]}

        return post_batch, select_action, next_game_status, actions

    def initial_action(self, batch, special_tokens_mask, seq_length, batch_max_seq_length, lm_device):
        batch_size = len(seq_length)
        actions = torch.zeros((batch_size, batch_max_seq_length), device=lm_device)
        game_status = torch.ones((batch_size, batch_max_seq_length))
        for i, index in enumerate(seq_length):
            game_status[i, index:] = 0
        action_value = torch.zeros((batch_size, batch_max_seq_length))
        return batch, actions, game_status, action_value
