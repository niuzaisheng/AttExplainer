from typing import Dict, List
import numpy as np
import random
from collections import namedtuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from accelerate.utils import send_to_device
import torch.optim as optim
from per import Memory
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

class DQNNet1D(nn.Module):

    def __init__(self, conig):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x, s, seq_len):
        batch_size, _ , max_seq_len= x.size()
        x = torch.cat([x, s.unsqueeze(1)], dim=1)  # [B, 2, seq]
        x = x.transpose(1, 2)  # [B, seq, 2]
        x = self.layer(x)
        return x.reshape(batch_size, max_seq_len)  # [B , seq]


class DQNNet4Grad(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(768, 8)
        self.out = nn.Linear(8+1, 1)

    def forward(self, x, s, seq_len):
        batch_size, _, max_seq_len, dim = x.size() # [B, seq, 1, dim]
        x = self.layer1(x[:,0]) # [B, seq, 8]
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
        x = self.out(x)  # [B, seq, 1]
        return x.reshape(batch_size, max_seq_len)  # [B , seq]


def gatherND(tensors: List[Dict[str, Tensor]], N=2)->Dict[str, Tensor]:
    """
        Gathers tensors from list of dict into dict. 
        N is the number of features dimensions to gather.
    """
    out_dict = {}
    first = tensors[0]
    batch_size = len(tensors)
    for k in first.keys():
        if "features" in k or "attentions" in k or "observation" in k:
            max_seq_len = max([item[k].size(1) for item in tensors])  # 2D [batch_size x seq x seq] or 1D [batch_size x seq]
            batch_features = []
            for i, item in enumerate(tensors):
                item_length = item[k].size(1)
                if N == 2:
                    temp = F.pad(item[k], (0, 0, 0, max_seq_len-item_length))
                elif N == 1:
                    temp = F.pad(item[k], (0, max_seq_len-item_length))
                else:
                    raise ValueError("Number of features dimensions for each example must be 1 or 2")
                batch_features.append(temp)
            batch_features = torch.stack(batch_features)
            out_dict[k] = batch_features
        else:
            if "done" in k or k in ["actions", "rewards"]:
                out_dict[k] = torch.stack([item[k] for item in tensors])
            elif k != "seq_length":
                out_dict[k] = pad_sequence([item[k] for item in tensors], batch_first=True)
    return out_dict


# def gather_rectangular(tensors: List[Dict[str, Tensor]])->Dict[str, Tensor]:
#     """
#         Gathers tensors from list of dict into dict. 
#         for feature matries, gather_rectangular() convert a list of [seq x dim] into [batch_size x seq x dim]
#     """
#     out_dict = {}
#     first = tensors[0]
#     batch_size = len(tensors)
#     for k in first.keys():
#         if "features" in k or "attentions" in k or "observation" in k:
#             max_seq_len = max([item[k].size(1) for item in tensors])  # [seq x dim] 
#             batch_features = []
#             for i, item in enumerate(tensors):
#                 item_length = item[k].size(1)
#                 if N == 2:
#                     temp = F.pad(item[k], (0, 0, 0, max_seq_len-item_length))
#                 elif N == 1:
#                     temp = F.pad(item[k], (0, max_seq_len-item_length))
#                 else:
#                     raise ValueError("Number of features dimensions for each example must be 1 or 2")
#                 batch_features.append(temp)
#             batch_features = torch.stack(batch_features)
#             out_dict[k] = batch_features
#         else:
#             if "done" in k or k in ["actions", "rewards"]:
#                 out_dict[k] = torch.stack([item[k] for item in tensors])
#             elif k != "seq_length":
#                 out_dict[k] = pad_sequence([item[k] for item in tensors], batch_first=True)
#     return out_dict



BufferItem = namedtuple("BufferItem", ("now_features", "next_features",
                                       "actions", "game_status", "next_game_status",
                                       "seq_length", "rewards", "ifdone"))


class DQN(object):
    """
        DQN progress for training
    """

    def __init__(self, config, mask_token_id=103, input_feature_shape=2):

        if config.train_agent_on_GPU:
            self.device = torch.device("cuda", config.gpu_index)
        else:
            self.device = torch.device("cpu")
        self.mask_token_id = mask_token_id
        self.input_feature_shape = input_feature_shape

        if config.features_type == "gradient":
            self.eval_net = DQNNet4Grad(config)
            self.target_net = DQNNet4Grad(config)
        else:
            if input_feature_shape == 2:
                self.eval_net = DQNNet(config)
                self.target_net = DQNNet(config)
            elif input_feature_shape == 1:
                self.eval_net = DQNNet1D(config)
                self.target_net = DQNNet1D(config)
        
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

    def choose_action(self, batch, batch_seq_length, special_tokens_mask, now_features, game_status, no_random=False):
        target_device = batch["input_ids"].device
        self.eval_net.eval()
        with torch.no_grad():
            actions = self.eval_net(now_features, game_status, batch_seq_length).detach()
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

    def store_transition(self, batch_size, now_features, next_features, game_status, next_game_status, actions, batch_seq_length, rewards, ifdone):
        self.eval_net.eval()
        with torch.no_grad():
            q_loss = self.get_td_loss(batch_seq_length, now_features, next_features, game_status, next_game_status, actions, rewards, ifdone).detach()

        now_features = send_to_device(now_features, "cpu")
        next_features = send_to_device(next_features, "cpu")
        game_status = send_to_device(game_status, "cpu")
        next_game_status = send_to_device(next_game_status, "cpu")
        actions = send_to_device(actions, "cpu")
        rewards = send_to_device(rewards, "cpu")
        ifdone = send_to_device(ifdone, "cpu")
        q_loss = send_to_device(q_loss, "cpu").numpy()

        temp_list = []
        for i in range(batch_size):
            new_item = BufferItem(now_features=now_features[i],
                                  next_features=next_features[i],
                                  actions=actions[i],
                                  game_status=game_status[i],
                                  next_game_status=next_game_status[i],
                                  seq_length=batch_seq_length[i],
                                  rewards=rewards[i],
                                  ifdone=ifdone[i],
                                  )
            temp_list.append(new_item)
            self.memory.add(q_loss[i], temp_list[i])

    def get_td_loss(self, batch_seq_length, now_features, next_features, game_status, next_game_status, actions, rewards, ifdone):
        ifdone = ifdone.float()
        q_eval = self.eval_net(now_features, game_status, batch_seq_length)
        max_seq_len = q_eval.size(1)
        actions = F.one_hot(actions, num_classes=max_seq_len).bool()

        q_eval = q_eval.masked_select(actions)  # [B, seq, 1]  -> [B, 1]

        with torch.no_grad():
            q_target = self.target_net(next_features, next_game_status, batch_seq_length)
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
        samples = gatherND(samples, self.input_feature_shape)
        samples = send_to_device(samples, self.device)
        now_features = samples["now_features"]
        next_features = samples["next_features"]
        game_status = samples["game_status"]
        next_game_status = samples["next_game_status"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        ifdone = samples["ifdone"]

        self.eval_net.train()
        q_loss = self.get_td_loss(batch_seq_length, now_features, next_features, game_status, next_game_status, actions, rewards, ifdone)

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

    def __init__(self, config, mask_token_id=103, input_feature_shape=2):

        self.device = torch.device("cpu")
        if config.features_type == "gradient":
            self.eval_net = DQNNet4Grad(config)
        else:
            if input_feature_shape == 2:
                self.eval_net = DQNNet(config)
            elif input_feature_shape == 1:
                self.eval_net = DQNNet1D(config)

        self.mask_token_id = mask_token_id
        with open(config.dqn_weights_path, "rb") as f:
            self.eval_net.load_state_dict(torch.load(f, map_location=self.device))

    def choose_action_for_eval(self, batch, batch_seq_length, special_tokens_mask, now_features, game_status):
        target_device = batch["input_ids"].device
        self.eval_net.eval()
        now_features = send_to_device(now_features, self.device)
        with torch.no_grad():
            actions = self.eval_net(now_features, game_status, batch_seq_length).detach()
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
