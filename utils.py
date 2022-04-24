from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import send_to_device
from torch import Tensor


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
                            simulate_batch_size: int, seq_length: List,
                            special_tokens_mask: Tensor,
                            simulate_batch: Dict[str, Tensor],
                            original_loss: Tensor,
                            original_pred_labels: Tensor,
                            original_prob: Tensor = None,
                            ):

    left_index = [i for i in range(simulate_batch_size) if original_acc[i].item() == 1]
    left_seq_length = [value for i, value in enumerate(seq_length) if i in left_index]

    left_original_acc = original_acc[left_index]
    left_special_tokens_mask = special_tokens_mask[left_index]

    left_simulate_batch = {}
    for key in simulate_batch.keys():
        left_simulate_batch[key] = simulate_batch[key][left_index]

    left_original_loss = original_loss[left_index]
    left_original_pred_labels = original_pred_labels[left_index]
    left_original_prob = original_prob[left_index]

    return len(left_index), left_seq_length, left_special_tokens_mask, left_simulate_batch, \
        left_original_loss, left_original_acc, left_original_pred_labels, left_original_prob


def display_ids(input_ids, tokenizer, name):
    input_ids = input_ids.cpu()
    string = tokenizer.decode(input_ids)
    string = string.replace("[PAD] ", "").replace("[PAD]", "")
    return string


def save_result(save_batch_size, completed_steps, original_input_ids, post_input_ids,
                golden_labels, original_pred_labels, post_pred_labels, delta_p, tokenizer, label_names, wandb_result_table):
    for i in range(save_batch_size):
        golden_label = golden_labels[i].item()
        golden_label = label_names[golden_label]
        original_pred_label = original_pred_labels[i].item()
        original_pred_label = label_names[original_pred_label]
        post_pred_label = post_pred_labels[i].item()
        post_pred_label = label_names[post_pred_label]
        item_delta_p = delta_p[i].item()
        train_batch_input_ids_example = display_ids(original_input_ids[i], tokenizer, name="train_batch_input_ids")
        post_batch_input_ids_example = display_ids(post_input_ids[i], tokenizer, name="post_batch_input_ids")
        wandb_result_table.add_data(completed_steps, golden_label, original_pred_label, post_pred_label, item_delta_p, train_batch_input_ids_example, post_batch_input_ids_example)

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
    # attentions = torch.sqrt(attentions) # do √x scaling

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
    y_pred = model_output.logits.detach().argmax(1).unsqueeze(-1)
    prob = torch.softmax(model_output.logits.detach(), dim=-1)
    prob = torch.gather(prob, 1, y_pred).reshape(-1)
    y_ref = y_ref.to(y_pred.device)
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
    y_pred = model_output.logits.detach().argmax(1).unsqueeze(-1)
    y_ref = y_ref.to(y_pred.device)
    prob = torch.softmax(model_output.logits.detach(), dim=-1)
    prob = torch.gather(prob, 1, y_ref).reshape(-1)
    accuracy = (y_ref == y_pred).float().reshape(-1)
    if device is not None:
        accuracy = accuracy.to(device)
        y_pred = y_pred.to(device)
        prob = prob.to(device)
    return accuracy, y_pred, prob


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


def compute_fidelity(original_model, finished_index, simulate_batch, special_tokens_mask,
                     game_status, original_pred_labels, lm_device, mask_token_id=103):

    fidelity_plus_batch = {}
    fidelity_minus_batch = {}
    for key in simulate_batch.keys():
        fidelity_plus_batch[key] = simulate_batch[key][finished_index]
        fidelity_minus_batch[key] = simulate_batch[key][finished_index]

    finish_game_status = game_status[finished_index]
    finish_special_tokens_mask = special_tokens_mask[finished_index]
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


def gather_unfinished_examples(ifdone: Tensor, simulate_batch_size: int, seq_length: List,
                               golden_labels: Tensor,
                               special_tokens_mask: Tensor,
                               next_attentions: Tensor,
                               now_game_status: Tensor,
                               simulate_batch: Dict[str, Tensor],
                               original_pred_labels: Tensor,
                               token_word_position_map: Dict[int, int],
                               cumulative_rewards: Tensor = None,
                               original_acc: Tensor = None,
                               original_loss: Tensor = None,
                               original_prob: Tensor = None,
                               delta_p: Tensor = None,
                               musked_token_rate: Tensor = None,
                               unmusked_token_rate: Tensor = None,
                               ):

    left_index = [i for i in range(simulate_batch_size) if ifdone[i].item() == 0]
    removed_index = [i for i in range(simulate_batch_size) if ifdone[i].item() == 1]
    left_seq_length = [value for i, value in enumerate(seq_length) if i in left_index]
    left_golden_labels = golden_labels[left_index]
    left_next_attentions = next_attentions[left_index]
    left_next_game_status = now_game_status[left_index]
    left_original_pred_labels = original_pred_labels[left_index]
    left_original_acc = original_acc[left_index]
    left_original_loss = original_loss[left_index]
    left_original_prob = original_prob[left_index]
    left_special_tokens_mask = special_tokens_mask[left_index]
    left_token_word_position_map = [v for i, v in enumerate(token_word_position_map) if i in left_index]
    left_cumulative_rewards = cumulative_rewards[left_index]

    next_simulate_batch = {}
    for key in simulate_batch.keys():
        next_simulate_batch[key] = simulate_batch[key][left_index]

    if delta_p is not None:
        left_delta_p = delta_p[left_index]
        left_musked_token_rate = musked_token_rate[left_index]
        left_unmusked_token_rate = unmusked_token_rate[left_index]

        return len(left_index), left_seq_length, left_golden_labels, left_special_tokens_mask, \
            left_next_attentions, left_next_game_status, next_simulate_batch, \
            left_original_pred_labels, left_token_word_position_map, left_cumulative_rewards, \
            left_original_acc, left_original_loss, left_original_prob, \
            left_delta_p, left_musked_token_rate, left_unmusked_token_rate, \
            removed_index
    else:
        return len(left_index), left_seq_length, left_golden_labels, left_special_tokens_mask, \
            left_next_attentions, left_next_game_status, next_simulate_batch, \
            left_original_pred_labels, left_token_word_position_map, left_cumulative_rewards, \
            left_original_acc, left_original_loss, left_original_prob, \
            removed_index