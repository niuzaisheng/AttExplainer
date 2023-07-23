from typing import Dict, List, NamedTuple
from collections import defaultdict
import logging

import numpy as np
import torch
import torch.nn as nn
from accelerate.utils import send_to_device
from torch import Tensor
from sklearn.metrics import auc

from functools import partial
from ei_net import modify_effective_information

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape_dict = {
    "const": 2,
    "random": 2,
    "input_ids": 1,
    "original_embedding": 2,
    "statistical_bin": 2,
    "effective_information": 1,
    "gradient": 2,
    "gradient_input": 2,
    "mixture": 2,
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
    # statistical_bin
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


def layer_forward_hook(module, hook_inputs, hook_outputs, embedding_saver):
    embedding_saver.append(hook_outputs)


def keep_tensor_in_same_device(*args, device):
    return [arg.to(device) for arg in args]


def get_gradient_features(transformer_model, post_batch, original_pred_labels, times_input=False):

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
        if times_input:
            grads = grads * input_embedding  # grad * input_embedding
    return grads.detach(), model_outputs


def use_original_embedding_as_features(transformer_model, post_batch):
    embedding_layer = transformer_model.bert.embeddings
    embedding_saver = []
    with torch.no_grad():
        hook = embedding_layer.register_forward_hook(partial(layer_forward_hook, embedding_saver=embedding_saver))
        model_outputs = transformer_model(**post_batch)
        hook.remove()
        input_embedding = embedding_saver[0]
    return input_embedding.detach(), model_outputs

def get_mixture_features(transformer_model, post_batch, original_pred_labels, seq_length, bins_num):
    # mixture of original_embedding, statistical_bin, and gradient features
    transformer_model.zero_grad()
    original_pred_labels = original_pred_labels.view(-1)
    embedding_layer = transformer_model.bert.embeddings
    embedding_saver = []
    with torch.autograd.set_grad_enabled(True):
        hook = embedding_layer.register_forward_hook(partial(layer_forward_hook, embedding_saver=embedding_saver))
        model_outputs = transformer_model(**post_batch, output_attentions=True)
        logits = model_outputs.logits
        hook.remove()
        input_embedding = embedding_saver[0]
        label_logits = logits[torch.arange(logits.size(0), device=logits.device), original_pred_labels]
        grads = torch.autograd.grad(torch.unbind(label_logits), input_embedding)[0]  # [batch_size, seq_len, model_rep_dim]
    
    with torch.no_grad():
        embedding_features = input_embedding.detach() # [batch_size, seq_len, model_rep_dim]
        attention_features = get_attention_features(model_outputs, post_batch["attention_mask"], seq_length, bins_num) # [batch_size, seq_len, bins_num * 2 + 4]
        grads_features = grads.detach() # [batch_size, seq_len, model_rep_dim]

    # mixture of original_embedding, statistical_bin, and gradient features
    extracted_features = torch.cat([embedding_features, attention_features, grads_features], dim=-1) # [batch_size, seq_len, 2 * model_rep_dim + bins_num * 2 + 4]

    return extracted_features, model_outputs


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


modification_budget_thresholds = list(range(1,11))  # mask token number form 1 to 10

def compute_salient_desc_auc(transformer_model, tracker,
                         input_ids, token_type_ids, attention_mask, special_tokens_mask: Tensor,
                         lm_device, mask_token_id=103):
    assert input_ids.size(0) == 1

    # salient desc auc
    valid_token_num = tracker.valid_token_num
    token_saliency = tracker.token_saliency
    original_pred_label = tracker.original_pred_label
    original_seq_length = tracker.original_seq_length

    input_ids = input_ids[:, :original_seq_length]
    token_type_ids = token_type_ids[:, :original_seq_length]
    attention_mask = attention_mask[:, :original_seq_length]
    special_tokens_mask = special_tokens_mask[:, :original_seq_length]

    assert input_ids.size(1) == len(token_saliency)

    token_saliency = np.array(token_saliency)
    token_saliency[special_tokens_mask[0]] = -np.inf  # set the saliency of special tokens to 0.0, like [CLS], [SEP], [PAD].
    sorted_saliency = np.argsort(token_saliency)[::-1]
    sorted_saliency = np.ascontiguousarray(sorted_saliency)

    thresholds_pred_probs = []
    thresholds_mask_token_num = []
    thresholds_mask_ratio = []

    for threshold in modification_budget_thresholds:

        if valid_token_num >= threshold:
            mask_token_num = threshold
        else:
            break

        mask_ratio = mask_token_num / valid_token_num
        masked_input_ids = input_ids.clone()
        mask_map = torch.zeros_like(masked_input_ids)
        mask_map[0, sorted_saliency[0:mask_token_num]] = 1
        mask_map.masked_fill_(special_tokens_mask, 0)
        masked_input_ids.masked_fill_(mask_map, mask_token_id)
        masked_input_ids = send_to_device(masked_input_ids, lm_device)
        token_type_ids = send_to_device(token_type_ids, lm_device)
        attention_mask = send_to_device(attention_mask, lm_device)
        with torch.no_grad():
            outputs = transformer_model(input_ids=masked_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            pred_prob = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            pred_prob = pred_prob[0, original_pred_label].item()
        thresholds_pred_probs.append(pred_prob)
        thresholds_mask_token_num.append(mask_token_num)
        thresholds_mask_ratio.append(mask_ratio)

    tracker.salient_desc_metrics = {
        "pred_probs": thresholds_pred_probs,
        "mask_token_num": thresholds_mask_token_num,
        "mask_ratio": thresholds_mask_ratio,
    }


def compute_modified_order_auc(transformer_model, tracker,
                         input_ids, token_type_ids, attention_mask, special_tokens_mask: Tensor,
                         lm_device, mask_token_id=103):
    assert input_ids.size(0) == 1

    # modified order auc
    valid_token_num = tracker.valid_token_num
    modified_index_order = tracker.modified_index_order
    original_pred_label = tracker.original_pred_label
    original_seq_length = tracker.original_seq_length

    input_ids = input_ids[:, :original_seq_length]
    token_type_ids = token_type_ids[:, :original_seq_length]
    attention_mask = attention_mask[:, :original_seq_length]
    special_tokens_mask = special_tokens_mask[:, :original_seq_length]

    # remove modified_index_order where the token is special token
    modified_index_order = [i for i in modified_index_order if not special_tokens_mask[0, i]]
    modified_index_order = np.array(modified_index_order)

    thresholds_pred_probs = []
    thresholds_mask_token_num = []
    thresholds_mask_ratio = []

    for threshold in modification_budget_thresholds:

        if len(modified_index_order) >= threshold:
            mask_token_num = threshold
        else:
            break

        mask_ratio = mask_token_num / valid_token_num
        masked_input_ids = input_ids.clone()
        mask_map = torch.zeros_like(masked_input_ids)
        mask_map[0, modified_index_order[0:mask_token_num]] = 1
        mask_map.masked_fill_(special_tokens_mask, 0)
        masked_input_ids.masked_fill_(mask_map, mask_token_id)
        masked_input_ids = send_to_device(masked_input_ids, lm_device)
        token_type_ids = send_to_device(token_type_ids, lm_device)
        attention_mask = send_to_device(attention_mask, lm_device)
        with torch.no_grad():
            outputs = transformer_model(input_ids=masked_input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            pred_prob = torch.softmax(outputs.logits, dim=-1).detach().cpu()
            pred_prob = pred_prob[0, original_pred_label].item()
        thresholds_pred_probs.append(pred_prob)
        thresholds_mask_token_num.append(mask_token_num)
        thresholds_mask_ratio.append(mask_ratio)

    tracker.modified_order_metrics = {
        "pred_probs": thresholds_pred_probs,
        "mask_token_num": thresholds_mask_token_num,
        "mask_ratio": thresholds_mask_ratio,
    }


def compute_auc(x, y):
    assert len(x) == len(y)
    x = np.array(x)
    y = np.array(y)
    # mask_token_num & pred_probs auc
    sorted_index = np.argsort(x)
    auc_score = auc(x[sorted_index], y[sorted_index])

    return auc_score

def plot_mask_token_num_curve(all_pred_probs, all_mask_token_num, all_mask_ratio):
    assert len(all_pred_probs) == len(all_mask_token_num) == len(all_mask_ratio)

    # plot mask_token_num vs average pred_probs curve 
    #      and mask_ratio vs average pred_probs curve
    mask_token_num_pred_probs = defaultdict(list)
    mask_ratio_pred_probs = defaultdict(list)

    for i in range(len(all_pred_probs)):
        pred_probs = all_pred_probs[i]
        mask_token_num = all_mask_token_num[i]
        mask_ratio = all_mask_ratio[i]
        mask_token_num_pred_probs[mask_token_num].append(pred_probs)
        mask_ratio_pred_probs[mask_ratio].append(pred_probs)

    return {
        "mask_token_num_pred_probs" : mask_token_num_pred_probs,
        "mask_ratio_pred_probs" : mask_ratio_pred_probs
    }


"""
    We compute the auc score at step 1 to 10. Marked as @K.
"""
auc_steps = list(range(1, 11))
def get_salient_desc_auc(all_trackers):
    res = {
        "mask_token_num_pred_probs_auc_score" : {},
        "mask_ratio_pred_probs_auc_score" : {}
    }
    for auc_step in auc_steps:
        all_pred_probs, all_mask_token_num, all_mask_ratio = [], [], []
        for tracker in all_trackers:
            tracker_step = len(tracker.salient_desc_metrics["pred_probs"])
            if tracker_step > auc_step:
                tracker_step = auc_step
            all_pred_probs.extend([tracker.original_prob] + tracker.salient_desc_metrics["pred_probs"][:tracker_step])
            all_mask_token_num.extend([0.0, ] + tracker.salient_desc_metrics["mask_token_num"][:tracker_step])
            all_mask_ratio.extend([0.0, ] +tracker.salient_desc_metrics["mask_ratio"][:tracker_step])
        res["mask_token_num_pred_probs_auc_score"][auc_step] = compute_auc(all_mask_token_num, all_pred_probs)
        res["mask_ratio_pred_probs_auc_score"][auc_step] = compute_auc(all_mask_ratio, all_pred_probs)
    return res

def get_modified_order_auc(all_trackers):
    res = {
        "mask_token_num_pred_probs_auc_score" : {},
        "mask_ratio_pred_probs_auc_score" : {}
    }
    for auc_step in auc_steps:
        all_pred_probs, all_mask_token_num, all_mask_ratio = [], [], []
        for tracker in all_trackers:
            tracker_step = len(tracker.modified_order_metrics["pred_probs"])
            if tracker_step > auc_step:
                tracker_step = auc_step
            all_pred_probs.extend([tracker.original_prob] + tracker.modified_order_metrics["pred_probs"][:tracker_step])
            all_mask_token_num.extend([0.0, ] + tracker.modified_order_metrics["mask_token_num"][:tracker_step])
            all_mask_ratio.extend([0.0, ] + tracker.modified_order_metrics["mask_ratio"][:tracker_step])
        res["mask_token_num_pred_probs_auc_score"][auc_step] = compute_auc(all_mask_token_num, all_pred_probs)
        res["mask_ratio_pred_probs_auc_score"][auc_step] = compute_auc(all_mask_ratio, all_pred_probs)
    return res


def compute_fidelity_when_masked(original_model, finished_index, original_batch, special_tokens_mask,
                                 game_status, original_pred_labels, lm_device, mask_token_id=103):

    # In game_status, 1 is visible to the model, 0 is invisible to the model.
    fidelity_plus_batch = {}
    original_batch = send_to_device(original_batch, lm_device)
    for key in original_batch.keys():
        fidelity_plus_batch[key] = original_batch[key][finished_index].clone()

    finish_game_status = game_status[finished_index]
    finish_special_tokens_mask = special_tokens_mask[finished_index]
    finish_special_tokens_mask = send_to_device(finish_special_tokens_mask, finish_game_status.device)
    finish_original_pred_labels = original_pred_labels[finished_index]

    fidelity_plus_mask = ~finish_game_status.bool()
    fidelity_plus_mask = fidelity_plus_mask.masked_fill(finish_special_tokens_mask, False)
    fidelity_plus_mask = send_to_device(fidelity_plus_mask, lm_device)

    fidelity_plus_batch["input_ids"] = torch.masked_fill(fidelity_plus_batch["input_ids"], fidelity_plus_mask, mask_token_id)

    with torch.no_grad():
        fidelity_plus_outputs = original_model(**fidelity_plus_batch)

    fidelity_plus_acc, finish_pred_labels, _ = batch_reversed_accuracy(fidelity_plus_outputs, finish_original_pred_labels, device="cpu")

    return fidelity_plus_acc


def compute_fidelity_when_deleted(original_model, finished_index, deleted_batch, special_tokens_mask,
                                  game_status, original_pred_labels, lm_device):

    fidelity_plus_batch = {}
    for key in deleted_batch.keys():
        fidelity_plus_batch[key] = deleted_batch[key][finished_index].clone()
    fidelity_plus_batch = send_to_device(fidelity_plus_batch, lm_device)

    # finish_game_status = game_status[finished_index]
    # finish_special_tokens_mask = special_tokens_mask[finished_index]
    # finish_special_tokens_mask = send_to_device(finish_special_tokens_mask, finish_game_status.device)
    finish_original_pred_labels = original_pred_labels[finished_index]

    # fidelity_plus_mask = ~finish_game_status.bool()
    # fidelity_plus_mask = fidelity_plus_mask.masked_fill(finish_special_tokens_mask, False)
    # fidelity_plus_mask = send_to_device(fidelity_plus_mask, lm_device)

    # fidelity_minus_mask = finish_game_status.bool()
    # fidelity_minus_mask = fidelity_minus_mask.masked_fill(finish_special_tokens_mask, False)
    # fidelity_minus_mask = send_to_device(fidelity_minus_mask, lm_device)

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



class GameEnvironmentVariables(NamedTuple):

    rewards: Tensor
    if_done: Tensor
    if_success: Tensor
    delta_prob: Tensor
    masked_token_rate: Tensor
    unmasked_token_rate: Tensor
    masked_token_num: Tensor
    unmask_token_num: Tensor
    delta_logits: Tensor
    delta_loss: Tensor

removed_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

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
        self.masked_token_rate: List or float = []
        self.masked_token_num: List or int = []
        self._token_saliency = None

        self.done_step = None  # auto track done step
        # after done
        self.fidelity = None
        self.salient_desc_metrics = None
        self.modified_order_metrics = None
        self.post_input_ids = None
        self.post_pred_label = None
        self.post_game_status = None

    @property
    def valid_token_num(self):
        if self.token_quantity_correction == 0:
            logger.warning("Token quantity correction is 0, this may be wrong.")
        return self.original_seq_length - self.token_quantity_correction

    def update(self,
               action: int = None,
               if_done: bool= None,
               if_success: bool= None,
               rewards: float= None,  # step reward
               post_prob: float = None,
               post_pred_label: int = None,
               post_logits: float = None,
               post_loss: float = None,
               delta_prob: float = None,
               delta_logits: float = None,
               masked_token_rate: float = None,
               masked_token_num: int = None,
               **kwargs):

        if self.done_step is not None:
            raise Exception("Tracker is done, can't update")

        if action:
            if self.token_replacement_strategy == "delete":
                # in delete mode, delete index from origional_index and add to modified_index_order
                origional_index = self.left_index[action]
                self.modified_index_order.append(origional_index)
                self.left_index.remove(origional_index)

            elif self.token_replacement_strategy == "mask":
                # in mask mode, add index to modified_index_order
                origional_index = self.left_index[action]
                self.modified_index_order.append(origional_index)
            else:
                raise ValueError("token_replacement_strategy should be delete or mask")

        assert not (not if_done and if_success), "Can't be success but not done"
        self.if_done = if_done
        self.if_success = if_success
        self.rewards.append(rewards)
        self.prob.append(post_prob)
        self.post_pred_label = post_pred_label
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
        if self._token_saliency is not None:
            return self._token_saliency
        else:
            saliency = defaultdict(float)
            last_prob = self.original_prob
            for token_index, prob in zip(self.modified_index_order, self.prob):
                saliency[token_index] = last_prob - prob
                last_prob = prob
            return [saliency[i] for i in range(self.original_seq_length)]

    @property
    def word_masked_rate(self):
        # NOTE: Here is a rough way of estimation, because the mask occurs at token level, not word level. But here it is seen as breaking the whole word.
        # if self.post_game_status is None:
        #     logger.warning("post_game_status is None, can't calculate word_masked_rate!")
        #     return None
        # word_num = len(set(self.token_word_position_map.values())) - self.token_quantity_correction
        # masked_word_index = set()
        # for i in range(self.original_seq_length):
        #     if self.post_game_status[i] == 0:
        #         masked_word_index.add(self.token_word_position_map[i])
        # return len(masked_word_index) / word_num
        word_num = len(set(self.token_word_position_map.values())) - self.token_quantity_correction
        masked_word_index = set()
        for i in self.modified_index_order:
            masked_word_index.add(self.token_word_position_map[i])
        return len(masked_word_index) / word_num

    # def get_openattack_word_masked_rate(self, tokenizer):
    #     """
    #         Copy from OpenAttack 
    #         OpenAttack/metric/algorithms/modification.py
    #     """

    #     original_input = self.input_ids.tolist()
    #     adversarial_input = self.post_input_ids.tolist()
    #     original_input = tokenizer.convert_ids_to_tokens(original_input)
    #     adversarial_input = tokenizer.convert_ids_to_tokens(adversarial_input)
    #     original_input = [t for t in original_input if t not in removed_special_tokens]
    #     adversarial_input = [t for t in adversarial_input if t not in removed_special_tokens]
    #     va = original_input
    #     vb = adversarial_input
    #     ret = 0
    #     if len(va) != len(vb):
    #         ret = abs(len(va) - len(vb))
    #     mn_len = min(len(va), len(vb))
    #     va, vb = va[:mn_len], vb[:mn_len]
    #     for wordA, wordB in zip(va, vb):
    #         if wordA != wordB:
    #             ret += 1
    #     return ret / len(original_input)

    def edit_distance(self, tokenizer) -> int:
        """
            Modified from OpenAttack
            OpenAttack/metric/selectors/edit_distance.py
        """
        original_input = self.input_ids.tolist()
        adversarial_input = self.post_input_ids.tolist()
        original_input = tokenizer.convert_ids_to_tokens(original_input)
        adversarial_input = tokenizer.convert_ids_to_tokens(adversarial_input)
        original_input = [t for t in original_input if t not in removed_special_tokens]
        adversarial_input = [t for t in adversarial_input if t not in removed_special_tokens]
        a = original_input
        b = adversarial_input
        la = len(a)
        lb = len(b)
        f = torch.zeros(la + 1, lb + 1, dtype=torch.long)
        for i in range(la + 1):
            for j in range(lb + 1):
                if i == 0:
                    f[i][j] = j
                elif j == 0:
                    f[i][j] = i
                elif a[i - 1] == b[j - 1]:
                    f[i][j] = f[i - 1][j - 1]
                else:
                    f[i][j] = min(f[i - 1][j - 1], f[i - 1][j], f[i][j - 1]) + 1
        return f[la][lb].item()
    
    def set_token_saliency(self, token_saliency):
        if isinstance(token_saliency, np.ndarray):
            token_saliency = token_saliency.tolist()
        self._token_saliency = token_saliency

    def save_result_row(self, tokenizer, label_names, wandb_result_table, completed_steps=None):
        golden_label = label_names[self.golden_label]
        original_pred_label = label_names[self.original_pred_label]
        post_pred_label = label_names[self.post_pred_label]
        original_input_ids = display_ids(self.input_ids, tokenizer)
        post_input_ids = display_ids(self.post_input_ids, tokenizer)
        if completed_steps is not None:  # when train
            wandb_result_table.add_data(completed_steps, self.id, self.done_step, golden_label, original_pred_label, self.original_prob,
                                        post_pred_label, self.prob[-1], self.delta_prob[-1], self.masked_token_rate[-1], self.masked_token_num[-1],
                                        self.modified_index_order, self.prob, self.rewards, self.delta_prob, self.token_saliency,
                                        self.masked_token_rate, self.masked_token_num,
                                        self.fidelity, original_input_ids, post_input_ids)
        else:  # when eval
            wandb_result_table.add_data(self.id, self.done_step, golden_label, original_pred_label, self.original_prob,
                                        post_pred_label, self.prob[-1], self.delta_prob[-1], self.masked_token_rate[-1], self.masked_token_num[-1],
                                        self.modified_index_order, self.prob, self.rewards, self.delta_prob, self.token_saliency,
                                        self.masked_token_rate, self.masked_token_num,
                                        self.fidelity, original_input_ids, post_input_ids)



def create_result_table(mode="test"):
    import wandb
    table_columns = ["id", "done_step", "golden_label", "original_pred_label", "original_pred_prob",
                     "post_pred_label", "post_pred_prob", "delta_prob", "masked_token_rate", "masked_token_num",
                     "modified_index_order", "step_prob", "step_rewards", "step_delta_prob", "token_saliency",
                     "step_masked_token_rate", "step_masked_token_num",
                     "fidelity", "original_input_ids", "post_input_ids"]
    if mode == "train":
        table_columns = ["completed_steps", ] + table_columns
    return wandb.Table(columns=table_columns)


def create_trackers(ids, original_seq_length, input_ids, token_word_position_map,
                    golden_labels, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                    token_quantity_correction, token_replacement_strategy):

    golden_labels = golden_labels.view(-1).tolist()
    original_acc = original_acc.tolist()
    original_pred_labels = original_pred_labels.view(-1).tolist()
    original_prob = original_prob.tolist()
    original_logits = original_logits.tolist()
    original_loss = original_loss.tolist()

    trackers = []
    for i in range(len(ids)):
        trackers.append(TokenModifyTracker(ids[i], original_seq_length[i], input_ids[i], token_word_position_map[i],
                                           golden_labels[i], original_acc[i], original_pred_labels[i], original_prob[i], original_logits[i], original_loss[i],
                                           token_quantity_correction, token_replacement_strategy))
    return trackers


def update_trackers(trackers: List[TokenModifyTracker], variables: GameEnvironmentVariables = None, **kwargs):
    # all kwargs are tensor
    # convert all tensor to list
    dic = {}
    if variables:
        for key, value in variables._asdict().items():
            if value is not None:
                dic[key] = value.tolist()
        for key in kwargs.keys():
            dic[key] = kwargs[key].tolist()

    # update trackers
    for i in range(len(trackers)):
        trackers[i].update(**{key: dic[key][i] for key in dic.keys()})


def gather_unfinished_examples_with_tracker(if_done: Tensor,
                                            trackers: List[TokenModifyTracker],
                                            seq_length: List[int],
                                            original_seq_length: List[int], original_acc: Tensor, original_pred_labels: Tensor, original_prob: Tensor, original_logits: Tensor, original_loss: Tensor,
                                            special_tokens_mask: Tensor,
                                            features: Tensor,
                                            game_status: Tensor,
                                            batch: Dict[str, Tensor]):

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


def get_permutation_mask_matrix(seq_length: int, special_tokens_mask: List[bool] = None):
    num_special_tokens = 0
    if special_tokens_mask is not None:
        num_special_tokens = sum(special_tokens_mask)
    valid_num = seq_length - num_special_tokens
    valid_mask_matrix = np.zeros((2 ** valid_num, valid_num), dtype=np.int64)
    valid_mask_rate = np.zeros((2 ** valid_num), dtype=np.float32)
    for i in range(2 ** valid_num):
        valid_mask_matrix[i] = [int(x) for x in list(bin(i)[2:].zfill(valid_num))]
        valid_mask_rate[i] = sum(valid_mask_matrix[i]) / valid_num
    if special_tokens_mask is not None:
        mask_matrix = np.zeros((2 ** valid_num, seq_length), dtype=np.int64)
        i = 0
        for j, flag in enumerate(special_tokens_mask):
            if flag:
                mask_matrix[:, j] = 0
            else:
                mask_matrix[:, j] = valid_mask_matrix[:, i]
                i += 1
    else:
        mask_matrix = valid_mask_matrix
    return mask_matrix, valid_mask_rate


class StepTracker:
    def __init__(self):
        self.step = 0

    def __add__(self, other):
        self.step += other
        return self.step

    @property
    def current_step(self):
        return self.step
