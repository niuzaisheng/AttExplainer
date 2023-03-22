# Explainable baseline method captum
# You need install captum package first by following cmd:
#   pip install captum

import os
import sys
sys.path.append(os.getcwd()) # for import utils.py from parent directory, keep this line before import utils

import argparse
import logging

from collections import defaultdict
from functools import partial
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from accelerate.utils import send_to_device
from captum.attr import (FeatureAblation, KernelShap, LayerDeepLift,
                         LayerIntegratedGradients, Occlusion,
                         ShapleyValueSampling)
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data_utils import get_dataset_and_model, get_dataset_config
from utils import StepTracker, TokenModifyTracker, get_salient_desc_auc, create_result_table

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Explainable Baseline Toolkit Captum")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion, snli or sst2."
    )
    parser.add_argument("--explain_method", type=str, default=0,
                        help="One of FeatureAblation, Occlusion, KernelShap, ShapleyValueSampling, LIME, IntegratedGradients or DeepLift.")
    parser.add_argument("--max_sample_num", type=int, default=100)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--eval_test_batch_size", type=int, default=32, help="Batch size for eval. Only for LIME method.")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="attexplainer-dev")
    parser.add_argument("--discribe", type=str, default="Captum")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode, only run 10 examples.")

    args = parser.parse_args()
    return args

config = parse_args()
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger.info(f"Eval config: {config}")

dataset_config = get_dataset_config(config)
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]
token_quantity_correction = dataset_config["token_quantity_correction"]

if config.use_wandb:
    import wandb
    wandb.init(name=f"Explainer_{config.explain_method}_{config.data_set_name}_{config.max_sample_num}", project=config.wandb_project_name, config=config)
    wandb_result_table = create_result_table()

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

print("Start loading!")
transformer_model, train_dataset, eval_dataset = get_dataset_and_model(config, dataset_config, tokenizer)
print("Finish loading!")

device = torch.device("cuda", config.gpu_index)

transformer_model.eval()
transformer_model = send_to_device(transformer_model, device)

# To solve the problems posed by secial tokens
valid_ids_map = {}  # {valid_token_position: input_token_position}
valid_token_num = 0

# For compute fidelity auc
all_trackers = []

all_game_step_done_num = defaultdict(list)  # success rate in each game_step
all_game_step_mask_rate = defaultdict(list)  # token mask rate in each game_step
all_game_step_mask_token_num = defaultdict(list)  # mask token num in each game_step
all_done_step_musk_rate_score = defaultdict(list)  # token mask rate in each done example
all_done_step_musk_token_num = defaultdict(list)  # mask token num in each done example
all_done_step_unmusk_token_num = defaultdict(list)  # unmask token num in each done example

def predict(input_ids, token_type_ids=None, attention_mask=None, step_tracker=None, need_grad=False):
    
    input_ids = send_to_device(input_ids, device)
    token_type_ids = send_to_device(token_type_ids, device)
    attention_mask = send_to_device(attention_mask, device)
    
    if step_tracker:
        step_tracker += input_ids.size(0)
    if need_grad:
        output = transformer_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            output = transformer_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return output.logits.to("cpu")

def batch_predict(example_list, token_type_ids=None, attention_mask=None, input_special_token_ids=None, step_tracker=None):
    """
        predict a batch of examples, only for LIME.
    """
    token_type_ids = send_to_device(token_type_ids, device)
    attention_mask = send_to_device(attention_mask, device)
    
    eval_batch = len(example_list) // config.eval_test_batch_size + 1
    assert len(example_list) == config.max_sample_num
    if step_tracker:
        step_tracker += len(example_list)

    all_result = []
    for i in range(eval_batch):
        if (i+1) * config.eval_test_batch_size > len(example_list):
            small_batch = example_list[i * config.eval_test_batch_size:]
        else:
            small_batch = example_list[i * config.eval_test_batch_size:(i+1) * config.eval_test_batch_size]

        batch_size = len(small_batch)
        input_ids = torch.zeros((batch_size, token_type_ids.size(1)),dtype=torch.long)
        for j in range(batch_size):
            line = [int(token) for token in small_batch[j].split(" ")]            
            for k in range(valid_token_num):
                input_ids[j, valid_ids_map[k]] = line[k]

        input_ids = input_ids + input_special_token_ids
        input_ids = send_to_device(input_ids, device)

        outputs = transformer_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        predict_prob = outputs.logits.detach().cpu()
        predict_prob = torch.softmax(predict_prob, dim=-1)
        all_result.append(predict_prob)

    all_result = torch.cat(all_result, dim=0)
    return all_result.numpy()

if config.explain_method == "FeatureAblation":
    lig = FeatureAblation(predict)
elif config.explain_method == "Occlusion":
    lig = Occlusion(predict)
elif config.explain_method == "KernelShap":
    lig = KernelShap(predict)
elif config.explain_method == "ShapleyValueSampling":
    lig = ShapleyValueSampling(predict)
elif config.explain_method == "LIME":
    lig = LimeTextExplainer(bow=False, mask_string=str(tokenizer.mask_token_id))
elif config.explain_method == "IntegratedGradients":
    lig = LayerIntegratedGradients(predict, transformer_model.bert.embeddings)
elif config.explain_method == "DeepLift":
    lig = LayerDeepLift(transformer_model, transformer_model.bert.embeddings)

thresholds = list(range(1,11))
def compute_salient_desc_auc(valid_token_num, input_ids, token_type_ids, attention_mask, special_tokens_mask: Tensor,
    modified_index_order:List[float], original_pred_label, lm_device, mask_token_id=103):
    """
        A sentence example is masked sequentially by the order sort of token_saliency.
        Same logic as the `compute_salient_desc_auc` function in the file `utils.py`.
    """

    assert input_ids.size(0) == 1

    thresholds_pred_probs = []
    thresholds_mask_token_num = []
    thresholds_mask_ratio = []

    for threshold in thresholds:
        if valid_token_num >= threshold:
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
    
    return {
        "pred_probs": thresholds_pred_probs,
        "mask_token_num": thresholds_mask_token_num,
        "mask_ratio": thresholds_mask_ratio,
    }


for i, item in tqdm(enumerate(eval_dataset), total=len(eval_dataset), disable=config.disable_tqdm):

    if config.debug and i >= 10:
        break

    example_id = item["id"]

    if isinstance(text_col_name, str):
        encode = tokenizer.encode_plus(item[text_col_name],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    else:
        encode = tokenizer.encode_plus([item[text_col_name[0]], item[text_col_name[1]]],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    input_ids = encode.input_ids
    seq_length = input_ids.size(1)
    token_type_ids = encode.token_type_ids
    attention_mask = encode.attention_mask
    special_tokens_mask = encode.special_tokens_mask.bool()
    input_special_token_ids = input_ids.clone().masked_fill_(~special_tokens_mask, 0)
    input_special_token_ids = input_special_token_ids

    ref_input_ids = input_ids.clone().masked_fill_(~special_tokens_mask, MASK_TOKEN_ID)
    valid_token_num = torch.sum(special_tokens_mask == 0).item()

    golden_label = item["label"]

    predict_loogits = predict(input_ids, token_type_ids, attention_mask)[0]
    original_pred_label = torch.argmax(predict_loogits).item()
    original_pred_prob = torch.softmax(predict_loogits, 0)[original_pred_label].item()

    tracker = TokenModifyTracker(example_id, seq_length, input_ids[0], None,
                                 golden_label, None, original_pred_label, original_pred_prob, predict_loogits, None,
                                 token_quantity_correction, "mask")
    all_trackers.append(tracker)


    step_tracker = StepTracker()
    if config.explain_method == "FeatureAblation":
        summarize_res = lig.attribute(inputs=input_ids,
                                      target=original_pred_label,
                                      additional_forward_args=(token_type_ids, attention_mask, step_tracker))

    elif config.explain_method == "Occlusion":
        summarize_res = lig.attribute(inputs=input_ids,
                                      sliding_window_shapes=(3,),
                                      baselines=ref_input_ids,
                                      target=original_pred_label,
                                      additional_forward_args=(token_type_ids, attention_mask, step_tracker))

    elif config.explain_method == "KernelShap" or config.explain_method == "ShapleyValueSampling":
        summarize_res = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      target=original_pred_label,
                                      n_samples=config.max_sample_num,
                                      perturbations_per_eval=1,
                                      additional_forward_args=(token_type_ids, attention_mask, step_tracker))

    elif config.explain_method == "LIME":

        valid_input_ids = []
        valid_token_num = 0
        for i, token_id in enumerate(input_ids.tolist()[0]):
            if token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                valid_input_ids.append(token_id)
                valid_ids_map[valid_token_num] = i
                valid_token_num += 1
        valid_input_ids = " ".join([str(i) for i in valid_input_ids])

        lime_batch_predict = partial(batch_predict, token_type_ids=token_type_ids, attention_mask=attention_mask, input_special_token_ids=input_special_token_ids, step_tracker=step_tracker)
        exp = lig.explain_instance(valid_input_ids, lime_batch_predict, num_samples=config.max_sample_num, num_features=valid_token_num, labels=(original_pred_label,))
        exp = exp.as_map()
        summarize_res = torch.zeros(input_ids.size())
        for token_id, weight in exp[original_pred_label]:
            summarize_res[0, valid_ids_map[token_id]] = weight

    elif config.explain_method == "IntegratedGradients":
        summarize_res = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      target=original_pred_label,
                                      n_steps=config.max_sample_num,
                                      additional_forward_args=(token_type_ids, attention_mask, step_tracker, True))
        summarize_res = summarize_res.sum(dim=-1)
        summarize_res = summarize_res / torch.norm(summarize_res, dim=-1, keepdim=True)
        summarize_res = summarize_res.detach().cpu()

    elif config.explain_method == "DeepLift":
        # add hook to transformer_model
        def step_tracker_hook(module, inputs, outputs, step_tracker):
            step_tracker += outputs.logits.size(0)
            return outputs.logits

        forward_handle = transformer_model.register_forward_hook(partial(step_tracker_hook, step_tracker=step_tracker))

        summarize_res = lig.attribute(inputs=send_to_device(input_ids, device),
                                      baselines=send_to_device(ref_input_ids, device),
                                      target=original_pred_label,
                                      additional_forward_args=(send_to_device(attention_mask, device), send_to_device(token_type_ids, device)))

        # unregister hook
        forward_handle.remove()

        summarize_res = summarize_res.sum(dim=-1)
        summarize_res = summarize_res / torch.norm(summarize_res, dim=-1, keepdim=True)
        summarize_res = summarize_res.detach().cpu()

    else:
        raise NotImplementedError(f"explain method {config.explain_method} not implemented")

    # size of summarize_res is [1, seq_len]
    mask_result = summarize_res > 0
    token_saliency = summarize_res[0].detach().numpy()
    token_saliency[special_tokens_mask[0]] = -np.inf # set the saliency of special tokens to 0.0, like [CLS], [SEP], [PAD].
    tracker.set_token_saliency(token_saliency)
    modified_index_order = np.argsort(token_saliency)[::-1]
    modified_index_order = np.ascontiguousarray(modified_index_order).tolist()
    tracker.modified_index_order = modified_index_order

    tracker.salient_desc_metrics = compute_salient_desc_auc(valid_token_num, input_ids, token_type_ids, attention_mask, special_tokens_mask,
                                                            modified_index_order, original_pred_label, device)

# same function as in analysis_add_tracker.py
salient_desc_auc = get_salient_desc_auc(all_trackers)

# plot
reslut = {}
reslut["Eval Example Number"] = len(all_trackers)
reslut["Attack Success Rate"] = sum([item.if_success for item in all_trackers]) / len(all_trackers)
reslut["Token Modification Rate"] = np.mean([item.masked_token_rate for item in all_trackers])
reslut["Token Modification Number"] = np.mean([item.masked_token_num for item in all_trackers])
reslut["Average Model Query Times"] = np.mean([item.done_step for item in all_trackers])
reslut["Fidelity"] = sum([item.fidelity for item in all_trackers]) / len(all_trackers)
reslut["delta_prob"] = np.mean([item.delta_prob[-1] for item in all_trackers])

logger.info(f"Result")
for k, v in reslut.items():
    logger.info(f"{k}: {v}")

if config.use_wandb:
    for k, v in reslut.items():
        wandb.run.summary[k] = v

    # 1. salient_desc_auc mask_token_num_pred_probs_auc_score curve
    key = "mask_token_num_pred_probs_auc_score"
    data = [[step, value] for (step, value) in salient_desc_auc[key].items()]
    table = wandb.Table(data=data, columns=["step", "value"])
    wandb.log({f"salient_desc_{key}_curve": wandb.plot.scatter(table, "step", "value", title=f"salient_desc {key} Curve")})

    # 2. salient_desc_auc mask_ratio_pred_probs_auc_score curve
    key = "mask_ratio_pred_probs_auc_score"
    data = [[step, value] for (step, value) in salient_desc_auc[key].items()]
    table = wandb.Table(data=data, columns=["step", "value"])
    wandb.log({f"salient_desc_{key}_curve": wandb.plot.scatter(table, "step", "value", title=f"salient_desc {key} Curve")})

    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()
