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
        self.dropout = nn.Dropout(0.5)
        self.layer2 = nn.Linear(self.h2 + 1, self.h3)
        self.out = nn.Linear(self.h3, 1)

    def forward(self, x, s):
        batch_size, matrix_num, max_seq_len, _ = x.size()

        x = x.reshape(batch_size, matrix_num, max_seq_len, self.bins_num * 2 + 4)  # [B, matrix_num, seq, (bins_num * 2 + 4)]
        x = x.permute(0, 2, 3, 1)  # [Batch x seq x (bins_num+2) x matrix_num]
        x = x.reshape(batch_size, max_seq_len, self.h1)  # [B, matrix_num, seq, 8]
        x = self.layer1(x)  # [B, seq, bins_num x 4]
        x = torch.relu(x)
        x = self.dropout(x)
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, h2 + 1]
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x.reshape(batch_size, max_seq_len)  # [B , seq]


class DQNNet1D(nn.Module):

    def __init__(self, conig):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x, s):
        batch_size, _, max_seq_len = x.size()
        x = torch.cat([x, s.unsqueeze(1)], dim=1)  # [B, 2, seq]
        x = x.transpose(1, 2)  # [B, seq, 2]
        x = self.layer(x)
        return x.reshape(batch_size, max_seq_len)  # [B , seq]

class DQNNet2D(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(768, 8)
        self.out = nn.Linear(8+1, 1)

    def forward(self, x, s):
        batch_size, _, max_seq_len, dim = x.size()  # [B, seq, seq, dim]
        x = self.layer1(x[:, 0])  # [B, seq, 8]
        x = torch.relu(x)
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
        x = self.out(x)  # [B, seq, 1]
        return x.reshape(batch_size, max_seq_len)  # [B , seq]

class DQNNet4Grad(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.layer1 = nn.Linear(768, 8)
        self.out = nn.Linear(8+1, 1)

    def forward(self, x, s):
        batch_size, _, max_seq_len, dim = x.size()  # [B, seq, 1, dim]
        x = self.layer1(x[:, 0])  # [B, seq, 8]
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
        x = self.out(x)  # [B, seq, 1]
        return x.reshape(batch_size, max_seq_len)  # [B , seq]

class DQNNetEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, 128)
        self.layer1 = nn.Linear(128, 8)
        self.out = nn.Linear(8+1, 1)

    def forward(self, input_ids , s):
        batch_size, _, max_seq_len = input_ids.size()
        x = self.embedding(input_ids[:, 0])  # [B, seq, dim]
        x = self.layer1(x)  # [B, seq, 8]
        x = torch.relu(x)
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
        x = self.out(x)  # [B, seq, 1]
        return x.reshape(batch_size, max_seq_len)  # [B , seq]


# class DQNNetMixture(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # input: [batch_size, seq_len, 2 * model_rep_dim + bins_num * 2 + 4]
#         self.bins_num = config.bins_num
#         self.layer1 = nn.Linear(2 * 768 + self.bins_num * 2 + 4, 64)
#         self.layer2 = nn.Linear(64, 8)
#         self.out = nn.Linear(8+1, 1)

#     def forward(self, x, s):
#         batch_size, _, max_seq_len, dim = x.size()  # [B, 1, seq, dim]
#         x = x.reshape(batch_size, max_seq_len, dim)  # [B, seq, dim]
#         x = self.layer1(x)  # [B, seq, 64]
#         x = torch.relu(x)
#         x = self.layer2(x)  # [B, seq, 8]
#         x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
#         x = self.out(x)  # [B, seq, 1]
#         return x.reshape(batch_size, max_seq_len)  # [B , seq]

class DQNNetMixture(nn.Module):
    """
        Add dropout to the network
    """
    def __init__(self, config):
        super().__init__()
        # input: [batch_size, seq_len, 2 * model_rep_dim + bins_num * 2 + 4]
        self.bins_num = config.bins_num
        self.layer1 = nn.Linear( 2 * 768 + self.bins_num * 2 + 4, 64)
        self.layer2 = nn.Linear(64, 8)
        self.out = nn.Linear(8+1, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, s):
        # Add dropout to the network
        batch_size, _, max_seq_len, dim = x.size()  # [B, 1, seq, dim]
        x = x.reshape(batch_size, max_seq_len, dim)  # [B, seq, dim]
        x = self.layer1(x)  # [B, seq, 64]
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)  # [B, seq, 8]
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, 9]
        x = self.out(x)  # [B, seq, 1]
        return x.reshape(batch_size, max_seq_len)  # [B , seq]
    

def gatherND(tensors: List[Dict[str, Tensor]], N=2) -> Dict[str, Tensor]:
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
            if k in ["if_success", "actions", "rewards"]:
                out_dict[k] = torch.stack([item[k] for item in tensors])
            elif k != "seq_length":
                out_dict[k] = pad_sequence([item[k] for item in tensors], batch_first=True)
    return out_dict


BufferItem = namedtuple("BufferItem", ("now_special_tokens_mask", "next_special_tokens_mask",
                                       "now_features", "next_features",
                                       "actions", "game_status", "next_game_status",
                                       "seq_length", "rewards", "if_success"))


class DQN(object):
    """
        DQN progress for training
    """

    def __init__(self, config, do_eval=False, mask_token_id=103, input_feature_shape=2, replace_func=None):

        self.do_eval = do_eval
        self.device = torch.device("cuda", config.gpu_index) if config.is_agent_on_GPU else torch.device("cpu")
        self.mask_token_id = mask_token_id
        self.input_feature_shape = input_feature_shape
        self.token_replacement_strategy = config.token_replacement_strategy

        ModelClass = None
        if config.features_type in ["gradient", "gradient_input"]:
            ModelClass = DQNNet4Grad
        elif config.features_type == "original_embedding":
            ModelClass = DQNNet2D
        elif config.features_type == "input_ids":
            ModelClass = DQNNetEmbedding
        elif config.features_type == "mixture":
            ModelClass = DQNNetMixture
        else:
            if input_feature_shape == 2:
                ModelClass = DQNNet
            elif input_feature_shape == 1:
                ModelClass = DQNNet1D

        self.eval_net = ModelClass(config)
        if not do_eval:
            self.target_net = ModelClass(config)

        if do_eval:
            with open(config.dqn_weights_path, "rb") as f:
                self.eval_net.load_state_dict(torch.load(f, map_location=self.device))

        if not do_eval:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.optimizer = optim.Adam(self.eval_net.parameters(), lr=config.dqn_rl)

        self.eval_net = send_to_device(self.eval_net, self.device)
        if not do_eval:
            self.target_net = send_to_device(self.target_net, self.device)

        self.use_categorical_policy = config.use_categorical_policy
        if not do_eval:
            self.use_ddqn = config.use_ddqn
            self.learn_step_counter = 0
            self.max_memory_capacity = config.max_memory_capacity
            self.target_replace_iter = config.target_replace_iter
            self.dqn_batch_size = config.dqn_batch_size
            self.epsilon = config.epsilon
            self.gamma = config.gamma
            self.memory = Memory(self.max_memory_capacity)
            self.loss_func = nn.MSELoss(reduce=False)

    def select_action_by_policy(self, actions, batch_seq_length, special_tokens_mask, epsilon_greedy=False):
        actions = actions.masked_fill(special_tokens_mask.bool(), -np.inf)
        if self.use_categorical_policy:
            # categorical policy
            actions = torch.softmax(actions, dim=1)
            m = torch.distributions.categorical.Categorical(probs=actions)
            select_action = m.sample()
        else:
            # argmax policy
            select_action = torch.argmax(actions, dim=1)  # [B, seq_len] -> [B]
            for i, index in enumerate(batch_seq_length):
                if epsilon_greedy and np.random.uniform() > self.epsilon:
                    select_action[i] = random.randint(0, index-1)

        return select_action

    def choose_action(self, batch, batch_seq_length, special_tokens_mask, now_features, game_status, return_repeat_action_flag=False):
        target_device = batch["input_ids"].device
        self.eval_net.eval()
        now_features = send_to_device(now_features, self.device)
        game_status = send_to_device(game_status, self.device)
        special_tokens_mask = send_to_device(special_tokens_mask, self.device)
    
        with torch.no_grad():
            actions = self.eval_net(now_features, game_status).detach()

        select_action = self.select_action_by_policy(actions, batch_seq_length, special_tokens_mask, not self.do_eval)
        if return_repeat_action_flag: # if the agent choose a position that has been selected, the repeat_action_flag will be set to True
            repeat_action_flag = torch.zeros_like(select_action).bool()

        if self.token_replacement_strategy == "mask":
            next_game_status = game_status.clone()
            for i, position in enumerate(select_action): 
                if game_status[i, position] == 0: # filp the game status in select position
                    next_game_status[i, position] = 1
                    if return_repeat_action_flag: repeat_action_flag[i] = True
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
            next_special_tokens_mask = special_tokens_mask

        elif self.token_replacement_strategy == "delete":

            batch_size, last_max_seq_len = batch["input_ids"].shape
            post_input_ids = torch.zeros((batch_size, last_max_seq_len-1), dtype=torch.long, device=target_device)
            post_attention_mask = torch.zeros((batch_size, last_max_seq_len-1), dtype=torch.long, device=target_device)
            post_token_type_ids = torch.zeros((batch_size, last_max_seq_len-1), dtype=torch.long, device=target_device)
            next_special_tokens_mask = torch.zeros((batch_size, last_max_seq_len-1), dtype=torch.bool, device=target_device)

            for i, index in enumerate(batch_seq_length):
                select_action_index = select_action[i].item()
                post_input_ids[i, 0: select_action_index] = batch["input_ids"][i, 0: select_action_index]
                post_input_ids[i, select_action_index:] = batch["input_ids"][i, select_action_index+1:]
                post_attention_mask[i, 0: select_action_index] = batch["attention_mask"][i, 0: select_action_index]
                post_attention_mask[i, select_action_index:] = batch["attention_mask"][i, select_action_index+1:]
                post_token_type_ids[i, 0: select_action_index] = batch["token_type_ids"][i, 0: select_action_index]
                post_token_type_ids[i, select_action_index:] = batch["token_type_ids"][i, select_action_index+1:]
                next_special_tokens_mask[i, 0: select_action_index] = special_tokens_mask[i, 0: select_action_index]
                next_special_tokens_mask[i, select_action_index:] = special_tokens_mask[i, select_action_index+1:]

            post_batch = {"input_ids": post_input_ids, "attention_mask": post_attention_mask, "token_type_ids": post_token_type_ids}
            next_game_status = post_attention_mask

        if return_repeat_action_flag:
            return post_batch, select_action, next_game_status, next_special_tokens_mask, repeat_action_flag
        else:
            return post_batch, select_action, next_game_status, next_special_tokens_mask

    def initial_action(self, batch, special_tokens_mask, seq_length, batch_max_seq_length, device):
        batch_size = len(seq_length)
        actions = torch.zeros((batch_size, batch_max_seq_length))
        game_status = torch.ones((batch_size, batch_max_seq_length)) # 1 is visible to the model, 0 is invisible to the model.
        for i, index in enumerate(seq_length):
            game_status[i, index:] = 0
        return actions.to(device), game_status.to(device)

    def store_transition(self, batch_size, now_special_tokens_mask, next_special_tokens_mask, now_features, next_features, game_status, next_game_status, actions, batch_seq_length, rewards, if_success):
        if self.do_eval:
            raise Exception("Could not store transitions when do_eval!")

        self.eval_net.eval()
        with torch.no_grad():
            q_loss = self.get_td_loss(batch_seq_length, now_special_tokens_mask, next_special_tokens_mask, now_features, next_features, game_status, next_game_status, actions, rewards, if_success).detach()

        now_special_tokens_mask = send_to_device(now_special_tokens_mask, "cpu")
        next_special_tokens_mask = send_to_device(next_special_tokens_mask, "cpu")
        now_features = send_to_device(now_features, "cpu")
        next_features = send_to_device(next_features, "cpu")
        game_status = send_to_device(game_status, "cpu")
        next_game_status = send_to_device(next_game_status, "cpu")
        actions = send_to_device(actions, "cpu")
        rewards = send_to_device(rewards, "cpu")
        if_success = send_to_device(if_success, "cpu")
        q_loss = send_to_device(q_loss, "cpu").numpy()

        for i in range(batch_size):
            new_item = BufferItem(now_special_tokens_mask=now_special_tokens_mask[i],
                                  next_special_tokens_mask=next_special_tokens_mask[i],
                                  now_features=now_features[i],
                                  next_features=next_features[i],
                                  actions=actions[i],
                                  game_status=game_status[i],
                                  next_game_status=next_game_status[i],
                                  seq_length=batch_seq_length[i],
                                  rewards=rewards[i],
                                  if_success=if_success[i],
                                  )
            self.memory.add(q_loss[i], new_item)

    def get_td_loss(self, batch_seq_length, now_special_tokens_mask, next_special_tokens_mask, now_features, next_features, game_status, next_game_status, actions, rewards, if_success):
        if self.do_eval:
            raise Exception("Could not get_td_loss when do_eval!")
        if_success = if_success.float()
        now_features = send_to_device(now_features, self.device)
        game_status = send_to_device(game_status, self.device)
        q_eval = self.eval_net(now_features, game_status)
        q_eval = q_eval.gather(1, actions.unsqueeze(-1)).squeeze(-1)  # [B, seq, 1]  -> [B]

        with torch.no_grad():
            next_special_tokens_mask = send_to_device(next_special_tokens_mask, self.device)
            next_features = send_to_device(next_features, self.device)
            next_game_status = send_to_device(next_game_status, self.device)
            if self.use_ddqn:
                target_actions = self.eval_net(next_features, next_game_status)
                target_actions = self.select_action_by_policy(target_actions, batch_seq_length, next_special_tokens_mask, epsilon_greedy=False)
                q_target = self.target_net(next_features, next_game_status)
                q_target = q_target.gather(1, target_actions.unsqueeze(-1)).squeeze(-1)
            else:
                q_target = self.target_net(next_features, next_game_status)
                target_actions = self.select_action_by_policy(q_target, batch_seq_length, next_special_tokens_mask, epsilon_greedy=False)
                q_target = q_target.gather(1, target_actions.unsqueeze(-1)).squeeze(-1)  # [B, seq, 1]  -> [B]

        # q_target = q_target.masked_select(actions)
        q_target = rewards + self.gamma * (1 - if_success) * q_target.detach()

        q_loss = self.loss_func(q_eval, q_target)   # [B]

        return q_loss

    def learn(self):
        if self.do_eval:
            raise Exception("Could not learn when do_eval!")
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        samples, sample_idxs, is_weight = self.memory.sample(self.dqn_batch_size)
        is_weight = torch.FloatTensor(is_weight).reshape(self.dqn_batch_size, 1).to(self.device)
        samples = [item._asdict() for item in samples if isinstance(item, BufferItem)]
        valid_sample_num = len(samples)
        if valid_sample_num == 0:
            return 0
        batch_seq_length = [item["seq_length"] for item in samples]
        samples = gatherND(samples, self.input_feature_shape)
        samples = send_to_device(samples, self.device)
        now_special_tokens_mask = samples["now_special_tokens_mask"]
        next_special_tokens_mask = samples["next_special_tokens_mask"]
        now_features = samples["now_features"]
        next_features = samples["next_features"]
        game_status = samples["game_status"]
        next_game_status = samples["next_game_status"]
        actions = samples["actions"]
        rewards = samples["rewards"]
        if_success = samples["if_success"]

        self.eval_net.train()
        q_loss = self.get_td_loss(batch_seq_length, now_special_tokens_mask, next_special_tokens_mask, now_features, next_features, game_status, next_game_status, actions, rewards, if_success)

        q_loss_detached = q_loss.detach()
        for i in range(valid_sample_num):
            idx = sample_idxs[i]
            self.memory.update(idx, q_loss_detached[i].item())

        dqn_loss = q_loss * is_weight
        dqn_loss = dqn_loss.mean()
        dqn_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=5)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return dqn_loss.detach().item()
