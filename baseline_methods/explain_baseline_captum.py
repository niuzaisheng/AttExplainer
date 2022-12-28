# Explainable baseline method captum
# You need install captum package first by following cmd:
#   pip install captum

import os
import sys
sys.path.append(os.getcwd())

import argparse
import logging

from collections import defaultdict
from functools import partial
from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import torch
from accelerate.utils import send_to_device
from captum.attr import (FeatureAblation, KernelShap, LayerDeepLift,
                         LayerIntegratedGradients, Occlusion,
                         ShapleyValueSampling)
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from data_utils import get_dataset_and_model, get_dataset_config
from utils import StepTracker

logger = logging.getLogger(__name__)


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
    wandb.init(name=f"Explainer_{config.explain_method}_{config.max_sample_num}", project=config.wandb_project_name, config=config)
    table_columns = ["id", "done_step", "golden_label", "original_pred_label", "original_pred_prob",
                     "post_pred_label", "post_pred_prob", "delta_prob", "masked_token_rate", "masked_token_num",
                     "original_input_ids", "post_input_ids", "token_saliency"]
    wandb_result_table = wandb.Table(columns=table_columns)

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

# Metrics
attack_successful_num = 0  # Attack Success Rate
all_eval_example_num = 0  # Attack Success Rate
all_musked_word_rate = []
all_unmusked_token_rate = []
all_done_game_step = []  # Average Victim Model Query Times
all_fidelity = []
all_delta_prob = []

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
        output = transformer_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    else:
        with torch.no_grad():
            output = transformer_model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return output.logits

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

for item in tqdm(eval_dataset, total=len(eval_dataset), disable=config.disable_tqdm):

    example_id = item["id"]

    all_eval_example_num += 1
    if isinstance(text_col_name, str):
        encode = tokenizer.encode_plus(item[text_col_name],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    else:
        encode = tokenizer.encode_plus([item[text_col_name[0]], item[text_col_name[1]]],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    input_ids = encode.input_ids
    token_type_ids = encode.token_type_ids
    attention_mask = encode.attention_mask
    special_tokens_mask = encode.special_tokens_mask.bool()
    input_special_token_ids = input_ids.clone().masked_fill_(~special_tokens_mask, 0)
    input_special_token_ids = input_special_token_ids

    ref_input_ids = input_ids.clone().masked_fill_(~special_tokens_mask, MASK_TOKEN_ID)
    valid_token_num = torch.sum(special_tokens_mask == 0).item()

    golden_label = item["label"]

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    original_input_ids_text = " ".join(all_tokens)

    predict_loogits = predict(input_ids, token_type_ids, attention_mask)[0]
    original_pred_label = torch.argmax(predict_loogits).item()
    original_pred_prob = torch.softmax(predict_loogits, 0)[original_pred_label].item()

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

    elif config.explain_method == "DeepLift":
        
        # add hook to transformer_model
        def step_tracker_hook(module, inputs, outputs, step_tracker):
            step_tracker += outputs.logits.size(0)
            return outputs.logits

        forward_handle = transformer_model.register_forward_hook(partial(step_tracker_hook, step_tracker=step_tracker))

        summarize_res = lig.attribute(inputs=input_ids,
                                      baselines=ref_input_ids,
                                      target=original_pred_label,
                                      additional_forward_args=(attention_mask, token_type_ids))

        # unregister hook
        forward_handle.remove()

        summarize_res = summarize_res.sum(dim=-1)
        summarize_res = summarize_res / torch.norm(summarize_res, dim=-1, keepdim=True)

    else:
        raise NotImplementedError

    # size of summarize_res is [1, seq_len]
    mask_result = summarize_res > 0
    token_saliency = summarize_res[0].detach().tolist()

    post_input_ids = input_ids.clone()
    mask_result.masked_fill_(special_tokens_mask, False)
    post_input_ids.masked_fill_(mask_result, MASK_TOKEN_ID)
    post_input_ids_text = tokenizer.decode(post_input_ids[0])

    musk_token_num = torch.sum(post_input_ids == MASK_TOKEN_ID).item()
    unmusk_token_num = valid_token_num - musk_token_num
    mask_rate = musk_token_num / valid_token_num
    unmask_rate = 1 - mask_rate
    all_musked_word_rate.append(mask_rate)
    all_unmusked_token_rate.append(unmask_rate)

    predict_loogits = predict(post_input_ids, token_type_ids, attention_mask)[0]
    post_pred_label = torch.argmax(predict_loogits).item()
    post_pred_prob = torch.softmax(predict_loogits, 0)[original_pred_label].item()

    delta_prob = original_pred_prob - post_pred_prob
    all_delta_prob.append(delta_prob)

    golden_label_name = label_names[golden_label]
    original_label_name = label_names[original_pred_label]
    post_pred_label_name = label_names[post_pred_label]

    game_step = step_tracker.current_step

    if post_pred_label != original_pred_label:  # attack success
        attack_successful_num += 1
        all_done_game_step.append(game_step)
        all_done_step_musk_rate_score[game_step].append(mask_rate)
        all_done_step_musk_token_num[game_step].append(musk_token_num)
        all_done_step_unmusk_token_num[game_step].append(unmusk_token_num)
        all_fidelity.append(True)
        for i in range(config.max_sample_num):
            if i < game_step:
                all_game_step_done_num[i].append(False)
            else:
                all_game_step_done_num[i].append(True)
    else:
        all_fidelity.append(False)
        for i in range(config.max_sample_num):
            all_game_step_done_num[i].append(False)

    if config.use_wandb:
        wandb_result_table.add_data(example_id, game_step, golden_label_name, original_label_name, original_pred_prob,
                                    post_pred_label_name, post_pred_prob, delta_prob, mask_rate, musk_token_num,
                                    original_input_ids_text, post_input_ids_text, token_saliency)


reslut = {}
reslut["Eval Example Number"] = all_eval_example_num
reslut["Attack Success Rate"] = attack_successful_num / all_eval_example_num
reslut["Token Modification Rate"] = np.mean(all_musked_word_rate)
reslut["Token Left Rate"] = np.mean(all_unmusked_token_rate)
reslut["Average Model Query Times"] = np.mean(all_done_game_step)
reslut["Fidelity"] = np.mean(all_fidelity)
reslut["delta_prob"] = np.mean(all_delta_prob)

logger.info(f"Result")
for k, v in reslut.items():
    logger.info(f"{k}: {v}")

if config.use_wandb:
    for k, v in reslut.items():
        wandb.run.summary[k] = v

    data = [[step, np.mean(value_list)] for (step, value_list) in all_game_step_done_num.items()]
    table = wandb.Table(data=data, columns=["game_step", "attack_success_rate"])
    wandb.log({"trade_off": wandb.plot.line(table, "game_step", "attack_success_rate", title="Attack Success Rate & Game Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_game_step_mask_rate.items()]
    table2 = wandb.Table(data=data, columns=["game_step", "musked_rate"])
    wandb.log({"trade_off2": wandb.plot.scatter(table2, "game_step", "musked_rate", title="Musked Rate & Game Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_done_step_musk_rate_score.items()]
    table3 = wandb.Table(data=data, columns=["done_step", "musked_rate"])
    wandb.log({"trade_off3": wandb.plot.scatter(table3, "done_step", "musked_rate", title="Musked Rate & Done Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_game_step_mask_token_num.items()]
    table4 = wandb.Table(data=data, columns=["game_step", "mask_token_num"])
    wandb.log({"trade_off4": wandb.plot.line(table4, "game_step", "mask_token_num", title="Musked Token Number & Game Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_done_step_musk_token_num.items()]
    table5 = wandb.Table(data=data, columns=["done_step", "mask_token_num"])
    wandb.log({"trade_off5": wandb.plot.scatter(table5, "done_step", "mask_token_num", title="Musked Token Number & Done Step Trade Off")})

    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()
