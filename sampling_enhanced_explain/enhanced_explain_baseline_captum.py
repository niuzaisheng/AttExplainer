# Explainable baseline method captum
# You need install captum package first by following cmd:
#   pip install captum

import argparse
import logging
import os
import sys
from collections import defaultdict
from functools import partial

sys.path.append(os.getcwd())

import numpy as np
import torch
from accelerate.utils import send_to_device
from captum.attr import (FeatureAblation, FeaturePermutation, KernelShap,
                         Occlusion, ShapleyValueSampling)
from data_utils import get_dataset_and_model, get_dataset_config
from dqn_model import DQN
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from utils import input_feature_shape_dict

from sampling_enhanced_explain.enhanced_explain_model import EnhancedKernelShap


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Explainable Baseline Toolkit Captum")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion, snli or sst2."
    )
    parser.add_argument("--explain_method", type=str, default=0, help="One of FeatureAblation, Occlusion, KernelShap, ShapleyValueSampling")
    parser.add_argument("--max_sample_num", type=int, default=100)
    parser.add_argument("--gpu_index", type=int, default=0)

    # dqn model config
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--is_agent_on_GPU", type=bool, default=True)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["statistical_bin", "const", "random", "effective_information", "gradient"],)
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner-dev")
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
    wandb.init(name=f"Expaliner_{config.explain_method}_{config.max_sample_num}", project=config.wandb_project_name, config=config)
    table_columns = ["completed_steps", "method", "sample_label", "original_pred_label", "post_pred_label", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

print("Start loading!")
transformer_model, train_dataset, eval_dataset = get_dataset_and_model(config, dataset_config, tokenizer)
print("Finish loading!")

device = torch.device("cuda", config.gpu_index)

transformer_model.eval()
transformer_model = send_to_device(transformer_model, device)


class StepCounter:
    def __init__(self):
        self.step = 0

    def one_step(self):
        self.step += 1

    def get_step(self):
        return self.step


game_step_KernelShap = StepCounter()
game_step_EnhancedKernelShap = StepCounter()

# Metrics
metrics_dict = {
    "KernelShap": defaultdict(list),
    "EnhancedKernelShap": defaultdict(list),
}

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape = input_feature_shape_dict[config.features_type]
config.token_replacement_strategy = "mask"
dqn_model = DQN(config, do_eval=True, mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)


def predict(inputs, token_type_ids=None, attention_mask=None, game_step_counter: StepCounter = None):
    if game_step_counter:
        game_step_counter.one_step()
    with torch.no_grad():
        output = transformer_model(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return output.logits


if config.explain_method == "FeatureAblation":
    lig = FeatureAblation(predict)
elif config.explain_method == "Occlusion":
    lig = Occlusion(predict)
elif config.explain_method == "KernelShap":
    lig = EnhancedKernelShap(partial(predict, game_step_counter=game_step_EnhancedKernelShap),
                             transformer_model, dqn_model)
    original_lig = KernelShap(partial(predict, game_step_counter=game_step_KernelShap))
elif config.explain_method == "ShapleyValueSampling":
    lig = ShapleyValueSampling(predict)


def result_processing(method, input_ids, token_type_ids, attention_mask, special_tokens_mask, original_pred_label,
                      summarize_res, game_step_counter: StepCounter):

    mask_result = summarize_res > 0
    attack_input_ids = input_ids.clone()
    mask_result.masked_fill_(special_tokens_mask, False)
    attack_input_ids.masked_fill_(mask_result, MASK_TOKEN_ID)
    attack_text = tokenizer.decode(attack_input_ids[0])

    musk_token_num = torch.sum(attack_input_ids == MASK_TOKEN_ID).item()
    valid_token_num = torch.sum(special_tokens_mask == 0).item()
    unmusk_token_num = valid_token_num - musk_token_num
    mask_rate = musk_token_num / valid_token_num
    unmask_rate = 1 - mask_rate

    game_step = game_step_counter.get_step()

    metrics_dict[method]["musked_word_rate"].append(mask_rate)
    metrics_dict[method]["unmusked_token_rate"].append(unmask_rate)

    predict_loogits = predict(attack_input_ids, token_type_ids, attention_mask)[0]
    post_pred_label = torch.argmax(predict_loogits).item()
    post_pred_prob = torch.softmax(predict_loogits, 0)[original_pred_label].item()
    metrics_dict[method]["eval_num"].append(1)

    if post_pred_label != original_pred_label:  # attack/flip success
        metrics_dict[method]["successful_num"].append(1)
        metrics_dict[method]["done_game_step"].append(game_step)
        metrics_dict[method]["done_step_musk_rate"].append(mask_rate)
        metrics_dict[method]["done_step_musk_token_num"].append(musk_token_num)
        metrics_dict[method]["done_step_unmusk_token_num"].append(unmusk_token_num)
        metrics_dict[method]["fidelity"].append(True)

        golden_label_name = label_names[golden_label]
        original_pred_label_name = label_names[original_pred_label]
        post_pred_label_name = label_names[post_pred_label]
        if config.use_wandb:
            wandb_result_table.add_data(index, method, golden_label_name, original_pred_label_name, post_pred_label_name, train_batch_input_ids_example, attack_text)
    else:
        metrics_dict[method]["fidelity"].append(False)

    # delta prob
    metrics_dict[method]["delta_prob"].append(original_pred_prob - post_pred_prob)


for index, item in tqdm(enumerate(eval_dataset), total=len(eval_dataset), disable=config.disable_tqdm):

    if isinstance(text_col_name, str):
        encode = tokenizer.encode_plus(item[text_col_name],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    else:
        encode = tokenizer.encode_plus([item[text_col_name[0]], item[text_col_name[1]]],
                                       return_tensors='pt', return_token_type_ids=True, return_attention_mask=True, return_special_tokens_mask=True)
    encode = send_to_device(encode, device)
    input_ids = encode.input_ids
    token_type_ids = encode.token_type_ids
    attention_mask = encode.attention_mask
    special_tokens_mask = encode.special_tokens_mask.bool()

    ref_input_ids = input_ids.clone().masked_fill_(~special_tokens_mask, MASK_TOKEN_ID)
    

    golden_label = item["label"]

    indices = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(indices)
    train_batch_input_ids_example = " ".join(all_tokens)

    predict_loogits = predict(input_ids, token_type_ids, attention_mask)[0]
    original_pred_label = torch.argmax(predict_loogits).item()
    original_pred_prob = torch.softmax(predict_loogits, 0)[original_pred_label].item()

    if config.explain_method == "FeatureAblation":
        summarize_res = lig.attribute(inputs=input_ids,
                                      target=original_pred_label,
                                      additional_forward_args=(token_type_ids, attention_mask))

    elif config.explain_method == "Occlusion":
        summarize_res = lig.attribute(inputs=input_ids,
                                      sliding_window_shapes=(3,),
                                      baselines=ref_input_ids,
                                      target=original_pred_label,
                                      special_tokens_mask=special_tokens_mask,
                                      additional_forward_args=(token_type_ids, attention_mask))

    elif config.explain_method == "KernelShap" or config.explain_method == "ShapleyValueSampling":
        enhanced_summarize_res = lig.attribute(inputs=input_ids,
                                               baselines=ref_input_ids,
                                               target=original_pred_label,
                                               n_samples=config.max_sample_num,
                                               perturbations_per_eval=1,
                                               token_type_ids=token_type_ids,
                                               attention_mask=attention_mask,
                                               special_tokens_mask=special_tokens_mask,
                                               additional_forward_args=(token_type_ids, attention_mask))
        metrics_dict["EnhancedKernelShap"]["summarize_res"].extend(enhanced_summarize_res[0].cpu().tolist())
        result_processing("EnhancedKernelShap", input_ids, token_type_ids, attention_mask, special_tokens_mask, original_pred_label,
                          enhanced_summarize_res, game_step_EnhancedKernelShap)

        original_summarize_res = original_lig.attribute(inputs=input_ids,
                                                        baselines=ref_input_ids,
                                                        target=original_pred_label,
                                                        n_samples=config.max_sample_num,
                                                        perturbations_per_eval=1,
                                                        additional_forward_args=(token_type_ids, attention_mask))
        metrics_dict["KernelShap"]["summarize_res"].extend(original_summarize_res[0].cpu().tolist())
        result_processing("KernelShap", input_ids, token_type_ids, attention_mask, special_tokens_mask, original_pred_label,
                          original_summarize_res, game_step_KernelShap)

# compute the correlation between all_enhanced_summarize_res and all_original_summarize_res
reslut = {}
for method in metrics_dict.keys():
    reslut[method] = {}
    for metric in metrics_dict[method].keys():
        if metric == "summarize_res":
            continue
        elif metric in ["eval_num", "successful_num"]:
            reslut[method][metric] = np.sum(metrics_dict[method][metric])
        else:
            reslut[method][metric] = np.mean(metrics_dict[method][metric])
    reslut[method]["Attack Success Rate"] = reslut[method]["successful_num"] / reslut[method]["eval_num"]

reslut["corrcoef"] = np.corrcoef(metrics_dict["EnhancedKernelShap"]["summarize_res"], metrics_dict["KernelShap"]["summarize_res"])

logger.info(f"Result")
for k, v in reslut.items():
    logger.info(f"{k}: {v}")

if config.use_wandb:
    for k, v in reslut.items():
        wandb.run.summary[k] = v

    data = list(zip(metrics_dict["EnhancedKernelShap"]["summarize_res"], metrics_dict["KernelShap"]["summarize_res"]))
    table6 = wandb.Table(data=data, columns=["enhanced_summarize_value", "original_summarize_value"])
    wandb.log({"table6": wandb.plot.scatter(table6, "enhanced_summarize_value", "original_summarize_value", title="Enhanced & Original Summarize Value")})

    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()
