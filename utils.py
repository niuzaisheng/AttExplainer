from typing import Dict, List
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import send_to_device
from torch import Tensor

from functools import partial
from ei_net import modify_effective_information


# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape_dict = {
    "statistical_bin": 2,
    "const": 2,
    "random": 2,
    "effective_information": 1,
    "gradient": 2,
}


def get_attention(model_outputs, layer_sets=None):
    """
        Get attention metrix from the model. Output size is [Batch, 1, seq_len, seq_len]
    """
    batch_attentions = model_outputs.attentions
    if layer_sets == 1:
        batch_attentions = batch_attentions[0:5]
    elif layer_sets == 2:
        batch_attentions = batch_attentions[5:9]
    elif layer_sets == 3:
        batch_attentions = batch_attentions[9:12]
    attentions = torch.cat([torch.mean(layer, dim=1, keepdim=True) for layer in batch_attentions], dim=1)
    return attentions.mean(1, keepdim=True).detach()  # [Batch,1,seq,seq]


def gather_correct_examples(original_acc: Tensor,
                            batch_size: int, seq_length: List,
                            special_tokens_mask: Tensor,
                            batch: Dict[str, Tensor],
                            original_loss: Tensor,
                            original_pred_labels: Tensor,
                            original_prob: Tensor = None,
                            ):

    left_index = [i for i in range(batch_size) if original_acc[i].item() == 1]
    left_seq_length = [value for i, value in enumerate(seq_length) if i in left_index]

    left_original_acc = original_acc[left_index]
    left_special_tokens_mask = special_tokens_mask[left_index]

    left_batch = {}
    for key in batch.keys():
        left_batch[key] = batch[key][left_index]

    left_original_loss = original_loss[left_index]
    left_original_pred_labels = original_pred_labels[left_index]
    left_original_prob = original_prob[left_index]

    return len(left_index), left_seq_length, left_special_tokens_mask, left_batch, \
        left_original_loss, left_original_acc, left_original_pred_labels, left_original_prob


def display_ids(input_ids, tokenizer):
    input_ids = input_ids.cpu()
    string = tokenizer.decode(input_ids)
    string = string.replace("[PAD] ", "").replace("[PAD]", "")
    return string


def save_result(save_batch_size, completed_steps, original_input_ids, post_input_ids,
                golden_labels, original_pred_labels, post_pred_labels, delta_p, tokenizer, label_names, wandb_result_table):
    # TODO: Deprecated
    for i in range(save_batch_size):
        golden_label = golden_labels[i].item()
        golden_label = label_names[golden_label]
        original_pred_label = original_pred_labels[i].item()
        original_pred_label = label_names[original_pred_label]
        post_pred_label = post_pred_labels[i].item()
        post_pred_label = label_names[post_pred_label]
        item_delta_p = delta_p[i].item()
        original_batch_input_ids_example = display_ids(original_input_ids[i], tokenizer)
        post_input_ids_example = display_ids(post_input_ids[i], tokenizer)
        wandb_result_table.add_data(completed_steps, golden_label, original_pred_label, post_pred_label, item_delta_p, original_batch_input_ids_example, post_input_ids_example)


def ids2list(input_ids, tokenizer):
    input_ids = input_ids.cpu()
    l = tokenizer.convert_ids_to_tokens(input_ids)
    return l


def get_empty_batch(referance_batch, special_tokens_mask, mask_token_id=103):
    """
        Replace valid token with [MASK]
    """
    empty_batch = {}
    empty_input_ids = referance_batch["input_ids"].clone()
    special_tokens_mask = send_to_device(special_tokens_mask, empty_input_ids.device)
    empty_input_ids = empty_input_ids.masked_fill(~special_tokens_mask, mask_token_id)
    empty_batch["input_ids"] = empty_input_ids
    empty_batch["attention_mask"] = referance_batch["attention_mask"]
    empty_batch["token_type_ids"] = referance_batch["token_type_ids"]
    return empty_batch


def get_random_attention_features(model_outputs, bins_num):
    batch_attentions = model_outputs.attentions
    one_layer_attention = batch_attentions[0]
    target_device = one_layer_attention.device
    batch_size, _, max_seq_len, _ = one_layer_attention.size()
    return torch.rand((batch_size, max_seq_len, 2 * bins_num + 4), device=target_device)


def get_const_attention_features(model_outputs, bins_num):
    batch_attentions = model_outputs.attentions
    one_layer_attention = batch_attentions[0]
    target_device = one_layer_attention.device
    batch_size, _, max_seq_len, _ = one_layer_attention.size()
    return torch.ones((batch_size, max_seq_len, 2 * bins_num + 4), device=target_device)


def get_EI_attention_features(model_outputs, batch_seq_len):
    batch_attentions = model_outputs.attentions
    attentions = torch.cat([torch.mean(layer, dim=1, keepdim=True) for layer in batch_attentions], dim=1)
    attentions = attentions.mean(1).detach().cpu()
    batch_size, max_seq_len, _ = attentions.size()
    target_device = attentions.device

    # [batch_size, max_seq_len]
    batch_ei_features = torch.zeros((batch_size, max_seq_len), device=target_device)
    for i in range(batch_size):
        vaild_attention = attentions[i, :batch_seq_len[i], :batch_seq_len[i]].numpy()
        ei_features = modify_effective_information(vaild_attention)
        batch_ei_features[i, :batch_seq_len[i]] = torch.tensor(ei_features, device=target_device)

    return batch_ei_features


def get_attention_features(model_outputs, attention_mask, batch_seq_len, bins_num):
    batch_attentions = model_outputs.attentions
    attentions = torch.cat([torch.mean(layer, dim=1, keepdim=True) for layer in batch_attentions], dim=1)
    attentions = attentions.mean(1).detach()
    batch_size, max_seq_len, _ = attentions.size()

    target_device = attentions.device
    spans = torch.linspace(0, 1, bins_num + 1, device=target_device)
    zeros = torch.zeros_like(attentions, device=target_device)
    lx = []
    ly = []
    attention_mask = attention_mask.unsqueeze(-1)
    attention_mask_float = attention_mask.float()
    square_attention_mask = torch.bmm(attention_mask_float, attention_mask_float.transpose(1, 2)).bool()
    attentions = attentions.masked_fill(~square_attention_mask, -np.inf)

    attentions = torch.clamp(attentions, 0, 1)

    last = attentions.where(attentions < 0, zeros).bool()
    for i in spans[1:]:
        values = attentions.where(attentions < i, zeros).bool()
        item = torch.logical_xor(values, last)
        lx.append(item.sum(dim=1))
        ly.append(item.sum(dim=2))
        last = torch.logical_or(item, last)

    lx = torch.stack(lx)  # [ bins * batch * seq_length ]
    ly = torch.stack(ly)  # [ bins * batch * seq_length ]
    stack = torch.vstack([lx, ly])  # [ 2bins * batch * seq_length ]
    stack = stack.permute(1, 2, 0)  # [ batch * seq_length * 2bins ]
    stack = stack / torch.FloatTensor(batch_seq_len).view((batch_size, 1, 1)).to(target_device)  # [ batch * seq_length * 2bins ]

    statistics = []
    for i, seq_len in enumerate(batch_seq_len):
        linex = attentions[i, :, 0:seq_len]  # along the x-axis
        liney = attentions[i, 0:seq_len, :]  # along the y-axis
        histc = torch.stack([torch.mean(linex, dim=1), torch.std(linex, dim=1),
                             torch.mean(liney, dim=0), torch.std(liney, dim=0)])  # [ batch * 2bins * seq_length ]
        statistics.append(histc)

    statistics = torch.stack(statistics)  # [ batch * 4 * seq_length ]
    statistics = statistics.transpose(1, 2)  # [ batch * seq_length * 4 ]
    statistics = statistics.masked_fill(~attention_mask.bool(), 0)

    stack = torch.cat([stack, statistics], dim=-1)  # [ batch_size, max_seq_len, bins_num * 2 + 4]
    return stack


# def get_gradient_features(model_outputs, batch_seq_len, input_ids, embedding_weight_tensor):
#     batch_size = len(batch_seq_len)
#     seq_len = input_ids.size(-1)
#     model_rep_dim = embedding_weight_tensor.size(-1)

#     logits = model_outputs.logits
#     logits.backward(torch.ones(logits.size(), device=logits.device)) # 此处有问题，需要修改
#     grad = torch.zeros((batch_size, seq_len, model_rep_dim), requires_grad=False)
#     for i in range(batch_size):
#         example_grad = embedding_weight_tensor.grad[input_ids[i]]
#         grad[i] = example_grad * embedding_weight_tensor[input_ids[i]]

#     return grad.detach()  # [batch_size, seq_len, model_rep_dim]


def layer_forward_hook(module, hook_inputs, hook_outputs, embedding_saver):
    embedding_saver.append(hook_outputs)


def get_gradient_features(transformer_model, post_batch, original_pred_labels):

    transformer_model.zero_grad()
    original_pred_labels = original_pred_labels.view(-1)
    embedding_layer = transformer_model.bert.embeddings
    embedding_saver = []
    with torch.autograd.set_grad_enabled(True):
        hook = embedding_layer.register_forward_hook(partial(layer_forward_hook, embedding_saver=embedding_saver))
        model_outputs = transformer_model(**post_batch)
        logits = model_outputs.logits
        hook.remove()
        input_embedding = embedding_saver[0]
        label_logits = logits[torch.arange(logits.size(0), device=logits.device), original_pred_labels]
        grads = torch.autograd.grad(torch.unbind(label_logits), input_embedding)[0]  # [batch_size, seq_len, model_rep_dim]
        grads = grads * input_embedding  # grad * input_embedding
    return grads.detach(), model_outputs


def batch_loss(model_output, y_ref, num_labels, device=None):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(model_output.logits.view(-1, num_labels).to(device), y_ref.to(device).view(-1)).detach()
    if device is not None:
        loss = loss.to(device)
    return loss


def batch_initial_prob(model_output, y_ref, device=None):
    """
        Note here that, in initial state: 
        - `y_ref` is refer to `golden_label`.
        - `y_pred` is obtained by the model's predicted label at this point, 
          which is used as `original_pred_labels` in future game process.
        - And `prob` is used as `original_prob` for compute \Delta p,  `\Delta p = original_prob - post_prob`.
          The `prob` here not referes to the probility of `y_ref`, but referes to the probility of `y_pred`.
    """
    device = model_output.logits.device
    y_pred = model_output.logits.detach().argmax(1).unsqueeze(-1)
    prob = torch.softmax(model_output.logits.detach(), dim=-1)
    prob = torch.gather(prob, 1, y_pred).reshape(-1)
    y_ref = y_ref.to(device)
    accuracy = (y_ref == y_pred).float().reshape(-1)
    if device is not None:
        accuracy = accuracy.to(device)
        y_pred = y_pred.to(device)
        prob = prob.to(device)
    return accuracy, y_pred, prob


def batch_accuracy(model_output, y_ref, device=None):
    """
        Note here that, in the game process, `y_ref` is `original_pred_labels`.
        So we are interested in the probability change of the `original_pred_labels` label.
    """
    device = model_output.logits.device
    y_pred = model_output.logits.detach().argmax(1).unsqueeze(-1)
    y_ref = y_ref.to(device)
    prob = torch.softmax(model_output.logits.detach(), dim=-1)
    prob = torch.gather(prob, 1, y_ref).reshape(-1)
    accuracy = (y_ref == y_pred).float().reshape(-1)
    if device is not None:
        accuracy = accuracy.to(device)
        y_pred = y_pred.to(device)
        prob = prob.to(device)
    return accuracy, y_pred, prob


def batch_logits(model_output, y_ref, device=None):
    """
        Note here that, in the game process, `y_ref` is `original_pred_labels`.
        So we are interested in the logits change of the `original_pred_labels` label.
    """
    device = model_output.logits.device
    y_ref = y_ref.to(device)
    y_ref_logits = torch.gather(model_output.logits.detach(), 1, y_ref).reshape(-1)
    if device is not None:
        y_ref_logits = y_ref_logits.to(device)
    return y_ref_logits


def batch_reversed_accuracy(model_output, y_ref, device=None):
    y_pred = model_output.logits.detach().argmax(1).unsqueeze(-1)
    y_ref = y_ref.to(y_pred.device)
    prob = torch.softmax(model_output.logits.detach(), dim=-1)
    prob = torch.gather(prob, 1, y_ref).reshape(-1)
    accuracy = (y_ref != y_pred).float().reshape(-1)
    if device is not None:
        accuracy = accuracy.to(device)
        y_pred = y_pred.to(device)
        prob = prob.to(device)
    return accuracy, y_pred, prob


def compute_fidelity_when_masked(original_model, finished_index, batch, special_tokens_mask,
                                 game_status, original_pred_labels, lm_device, mask_token_id=103):

    fidelity_plus_batch = {}
    fidelity_minus_batch = {}
    for key in batch.keys():
        fidelity_plus_batch[key] = batch[key][finished_index]
        fidelity_minus_batch[key] = batch[key][finished_index]

    finish_game_status = game_status[finished_index]
    finish_special_tokens_mask = special_tokens_mask[finished_index]
    finish_special_tokens_mask = send_to_device(finish_special_tokens_mask, finish_game_status.device)
    finish_original_pred_labels = original_pred_labels[finished_index]

    fidelity_plus_mask = ~finish_game_status.bool()
    fidelity_plus_mask = fidelity_plus_mask.masked_fill(finish_special_tokens_mask, False)
    fidelity_plus_mask = send_to_device(fidelity_plus_mask, lm_device)

    fidelity_minus_mask = finish_game_status.bool()
    fidelity_minus_mask = fidelity_minus_mask.masked_fill(finish_special_tokens_mask, False)
    fidelity_minus_mask = send_to_device(fidelity_minus_mask, lm_device)
    fidelity_plus_batch["input_ids"] = torch.masked_fill(fidelity_plus_batch["input_ids"], fidelity_plus_mask, mask_token_id)
    fidelity_minus_batch["input_ids"] = torch.masked_fill(fidelity_minus_batch["input_ids"], fidelity_minus_mask, mask_token_id)

    with torch.no_grad():
        fidelity_plus_outputs = original_model(**fidelity_plus_batch)
        fidelity_minus_outputs = original_model(**fidelity_minus_batch)

    fidelity_plus_acc, finish_pred_labels, _ = batch_reversed_accuracy(fidelity_plus_outputs, finish_original_pred_labels, device="cpu")
    fidelity_minus_acc, finish_pred_labels, _ = batch_accuracy(fidelity_minus_outputs, finish_original_pred_labels, device="cpu")

    return fidelity_plus_acc, fidelity_minus_acc


def compute_fidelity_when_deleted(original_model, finished_index, batch, special_tokens_mask,
                                  game_status, original_pred_labels, lm_device):

    fidelity_plus_batch = {}
    for key in batch.keys():
        fidelity_plus_batch[key] = batch[key][finished_index]

    finish_game_status = game_status[finished_index]
    finish_special_tokens_mask = special_tokens_mask[finished_index]
    finish_special_tokens_mask = send_to_device(finish_special_tokens_mask, finish_game_status.device)
    finish_original_pred_labels = original_pred_labels[finished_index]

    fidelity_plus_mask = ~finish_game_status.bool()
    fidelity_plus_mask = fidelity_plus_mask.masked_fill(finish_special_tokens_mask, False)
    fidelity_plus_mask = send_to_device(fidelity_plus_mask, lm_device)

    fidelity_minus_mask = finish_game_status.bool()
    fidelity_minus_mask = fidelity_minus_mask.masked_fill(finish_special_tokens_mask, False)
    fidelity_minus_mask = send_to_device(fidelity_minus_mask, lm_device)

    with torch.no_grad():
        fidelity_plus_outputs = original_model(**fidelity_plus_batch)

    fidelity_plus_acc, finish_pred_labels, _ = batch_reversed_accuracy(fidelity_plus_outputs, finish_original_pred_labels, device="cpu")

    return fidelity_plus_acc


def gather_unfinished_examples(if_done: Tensor, batch_size: int, seq_length: List[int],
                               ids: List[str],
                               original_seq_length: Tensor,
                               golden_labels: Tensor,
                               next_special_tokens_mask: Tensor,
                               next_attentions: Tensor,
                               now_game_status: Tensor,
                               original_batch: Dict[str, Tensor],
                               batch: Dict[str, Tensor],
                               original_pred_labels: Tensor,
                               token_word_position_map: Dict[int, int],
                               cumulative_rewards: Tensor = None,
                               original_acc: Tensor = None,
                               original_loss: Tensor = None,
                               original_prob: Tensor = None,
                               delta_p: Tensor = None,
                               masked_token_rate: Tensor = None,
                               unmasked_token_rate: Tensor = None,
                               ):

    left_index = [i for i in range(batch_size) if if_done[i].item() == 0]
    left_seq_length = [value for i, value in enumerate(seq_length) if i in left_index]
    left_ids = [value for i, value in enumerate(ids) if i in left_index]
    left_original_seq_length = [value for i, value in enumerate(original_seq_length) if i in left_index]
    left_golden_labels = golden_labels[left_index]
    left_next_attentions = next_attentions[left_index]
    left_next_game_status = now_game_status[left_index]
    left_original_pred_labels = original_pred_labels[left_index]
    left_original_acc = original_acc[left_index]
    left_original_loss = original_loss[left_index]
    left_original_prob = original_prob[left_index]
    left_next_special_tokens_mask = next_special_tokens_mask[left_index]
    left_token_word_position_map = [v for i, v in enumerate(token_word_position_map) if i in left_index]
    left_cumulative_rewards = cumulative_rewards[left_index]

    next_batch = {}
    for key in batch.keys():
        next_batch[key] = batch[key][left_index]

    next_original_batch = {}
    for key in original_batch.keys():
        next_original_batch[key] = original_batch[key][left_index]

    if delta_p is not None:
        left_delta_p = delta_p[left_index]
        left_masked_token_rate = masked_token_rate[left_index]
        left_unmasked_token_rate = unmasked_token_rate[left_index]

        return len(left_index), left_seq_length, left_ids, left_original_seq_length, left_golden_labels, left_next_special_tokens_mask, \
            left_next_attentions, left_next_game_status, next_original_batch, next_batch, \
            left_original_pred_labels, left_token_word_position_map, left_cumulative_rewards, \
            left_original_acc, left_original_loss, left_original_prob, \
            left_delta_p, left_masked_token_rate, left_unmasked_token_rate
    else:
        return len(left_index), left_seq_length, left_ids, left_original_seq_length, left_golden_labels, left_next_special_tokens_mask, \
            left_next_attentions, left_next_game_status, next_original_batch, next_batch, \
            left_original_pred_labels, left_token_word_position_map, left_cumulative_rewards, \
            left_original_acc, left_original_loss, left_original_prob


def clone_batch(batch: Dict[str, Tensor]):
    """
    Clone a batch of inputs and targets.
    """
    cloned_batch = {}
    for key in batch.keys():
        cloned_batch[key] = batch[key].clone()
    return cloned_batch


def get_most_free_gpu_index():
    gpu_usage = []
    for i in range(torch.cuda.device_count()):
        gpu_usage.append(torch.cuda.memory_allocated(i))
    return gpu_usage.index(min(gpu_usage))


# TokenModifyTracker and assist tracking functions
class TokenModifyTracker:
    """
        Track token modification order, initialize with a list of token indices.
        at each step, given a index of token deleted or modified , the tracker will record.
        in the end, the tracker will return a list of token indices in the order of modification.
        and compute the number of tokens modified, the ratio of tokens modified.
    """

    def __init__(self,
                 id: str,
                 original_seq_length: int,
                 input_ids: Tensor,
                 token_word_position_map: Dict[int, int],
                 golden_label: int,
                 original_acc: float,
                 original_pred_label: int,
                 original_prob: float,
                 original_logits: float,
                 original_loss: float,
                 token_quantity_correction: int,
                 token_replacement_strategy: str):

        self.id = id
        self.original_seq_length = original_seq_length
        self.input_ids = input_ids
        self.token_word_position_map = token_word_position_map
        self.golden_label = golden_label
        self.original_acc = original_acc
        self.original_pred_label = original_pred_label
        self.original_prob = original_prob
        self.original_logits = original_logits
        self.original_loss = original_loss
        self.token_quantity_correction = token_quantity_correction
        self.token_replacement_strategy = token_replacement_strategy

        # update by each token modification
        self.left_index = [i for i in range(original_seq_length)]
        self.modified_index_order = []  # record the order of modified token index
        self.if_done = False
        self.if_success = False
        self.rewards = []
        self.prob = []
        self.logits = []
        self.loss = []
        self.delta_prob = []
        self.delta_logits = []
        self.masked_token_rate = []
        self.masked_token_num = []

        # auto track done step
        self.done_step = None
        self.fidelity = None
        self.post_input_ids = None

    def update(self,
               action: int,
               if_done: bool,
               if_success: bool,
               reward: float,  # step reward
               post_prob: float,
               post_logits: float,
               post_loss: float,
               delta_prob: float,
               delta_logits: float,
               masked_token_rate: float,
               masked_token_num: int):

        if self.done_step is not None:
            raise Exception("Tracker is done, can't update")

        if self.token_replacement_strategy == "delete":
            # in delete mode, delete index from origional_index and add to modified_index_order
            origional_index = self.left_index[action]
            self.modified_index_order.append(origional_index)
            self.left_index.remove(action)
        elif self.token_replacement_strategy == "mask":
            # in mask mode, add index to modified_index_order
            origional_index = self.left_index[action]
            self.modified_index_order.append(origional_index)
        else:
            raise ValueError("token_replacement_strategy should be delete or mask")

        assert not (not if_done and if_success), "Can't be success but not done"
        self.if_done = if_done
        self.if_success = if_success
        self.rewards.append(reward)
        self.prob.append(post_prob)
        self.logits.append(post_logits)
        self.loss.append(post_loss)
        self.delta_prob.append(delta_prob)
        self.delta_logits.append(delta_logits)
        self.masked_token_rate.append(masked_token_rate)
        self.masked_token_num.append(masked_token_num)

        if if_done or if_success:
            self.done_step = len(self.prob)

    def done_and_fail(self):
        self.done_step = len(self.prob)
        self.if_done = True
        self.if_success = False

    @property
    def token_saliency(self):
        saliency = defaultdict(0.0)
        last_prob = self.original_prob
        for token_index, prob in zip(self.modified_index_order, self.prob):
            saliency[token_index] = last_prob - prob
            last_prob = prob
        return [saliency[i] for i in range(self.original_seq_length)]

    def save_result_row(self, tokenizer, label_names, wandb_result_table):
        golden_label = label_names[self.golden_label]
        original_pred_label = label_names[self.original_pred_label]
        post_pred_label = label_names[self.post_pred_label]
        original_input_ids = display_ids(self.input_ids, tokenizer)
        post_input_ids = display_ids(self.post_input_ids, tokenizer)
        wandb_result_table.add_data(self.id, self.done_step, golden_label, original_pred_label, self.original_prob,
                                    post_pred_label, self.prob[-1], self.delta_prob[-1], self.masked_token_rate[-1], self.masked_token_num[-1],
                                    self.modified_index_order, self.prob, self.rewards, self.delta_prob, self.token_saliency,
                                    self.masked_token_rate, self.masked_token_num,
                                    self.fidelity, original_input_ids, post_input_ids)


def create_result_table():
    import wandb
    table_columns = ["id", "done_step", "golden_label", "original_pred_label", "original_pred_prob",
                     "post_pred_label", "post_pred_prob", "delta_p", "masked_token_rate", "masked_token_num",
                     "modified_index_order", "step_prob", "step_rewards", "step_delta_prob", "token_saliency",
                     "step_masked_token_rate", "step_masked_token_num",
                     "fidelity", "original_input_ids", "post_input_ids"]
    return wandb.Table(columns=table_columns)


def create_trackers(ids, original_seq_length, input_ids, token_word_position_map,
                    golden_labels, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                    token_quantity_correction, token_replacement_strategy):
    trackers = []
    for i in range(len(ids)):
        trackers.append(TokenModifyTracker(ids[i], original_seq_length[i], input_ids[i], token_word_position_map[i],
                                           golden_labels[i], original_acc[i], original_pred_labels[i], original_prob[i], original_logits[i], original_loss[i],
                                           token_quantity_correction, token_replacement_strategy))
    return trackers


def update_trackers(trackers: List[TokenModifyTracker], **kwargs):

    # all kwargs are tensor
    # convert all tensor to list
    for key in kwargs.keys():
        kwargs[key] = kwargs[key].tolist()

    # update trackers
    for i in range(len(trackers)):
        trackers[i].update(**{key: kwargs[key][i] for key in kwargs.keys()})


def gather_unfinished_examples_with_tracker(if_done: Tensor,
                                            trackers: List[TokenModifyTracker],
                                            seq_length: List[int],
                                            original_seq_length: List[int], original_acc: Tensor, original_pred_labels: Tensor, original_prob: Tensor, original_logits: Tensor, original_loss: Tensor,
                                            special_tokens_mask: Tensor,
                                            features: Tensor,
                                            game_status: Tensor,
                                            batch: Dict[str, Tensor]
                                            ):

    batch_size = len(trackers)
    left_index = [i for i in range(batch_size) if if_done[i].item() == 0]
    left_trackers = [trackers[i] for i in left_index]
    left_seq_length = [value for i, value in enumerate(seq_length) if i in left_index]
    left_original_seq_length = [value for i, value in enumerate(original_seq_length) if i in left_index]
    left_original_acc = original_acc[left_index]
    left_original_pred_labels = original_pred_labels[left_index]
    left_original_prob = original_prob[left_index]
    left_original_logits = original_logits[left_index]
    left_original_loss = original_loss[left_index]
    left_special_tokens_mask = special_tokens_mask[left_index]
    left_features = features[left_index]
    left_game_status = game_status[left_index]

    left_batch = {}
    for key in batch.keys():
        left_batch[key] = batch[key][left_index]

    return len(left_index), left_trackers, left_seq_length, \
        left_original_seq_length, left_original_acc, left_original_pred_labels, left_original_prob, left_original_logits, left_original_loss, \
        left_special_tokens_mask, left_features, left_game_status, left_batch
