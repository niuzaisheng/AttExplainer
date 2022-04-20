# Explainable baseline method LIME
# You need install lime package first by following cmd:
#   pip install lime
# https://github.com/marcotcr/lime

import os
import sys
sys.path.append(os.getcwd())
import logging
import argparse

from collections import defaultdict

import numpy as np
import torch
from accelerate.utils import send_to_device
from data_utils import get_dataset_config, get_dataset_and_model
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Run Explainable baseline method Anchor")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--simulate_batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=8)
    parser.add_argument("--max_sample_num", type=int, default=100)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--disable_tqdm", action="store_true", default=False)    
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner")
    parser.add_argument("--discribe", type=str, default="LIME")

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
text_col_num = dataset_config["text_col_num"]

if config.use_wandb:
    import wandb
    wandb.init(name=f"Expaliner_LIME_{config.max_sample_num}", project=config.wandb_project_name, config=config)
    wandb_config = wandb.config
    table_columns = ["completed_steps", "sample_label", "original_pred_label", "post_pred_label", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

print("Start loading!")
transformer_model, train_dataset, eval_dataset = get_dataset_and_model(config, dataset_config, tokenizer)
print("Finish loading!")

device = torch.device("cuda", config.gpu_index)

transformer_model.eval()
transformer_model = send_to_device(transformer_model, device)


game_step = 0
# To solve the problems posed by secial tokens
valid_ids_map = {}  # valid_token_position: input_token_position
valid_token_num = 0
input_special_token_ids = None
input_token_type_ids = None

# Metrics
attack_successful_num = 0  # Attack Success Rate
all_eval_example_num = 0  # Attack Success Rate
all_musked_token_rate = []
all_unmusked_token_rate = []
# all_done_game_step = []  # Average Victim Model Query Times
fidelity_add = []
all_delta_prob = []


all_game_step_mask_rate = defaultdict(list) # token mask rate in each game_step
all_game_step_mask_token_num = defaultdict(list) # mask token num in each game_step
all_done_step_musk_rate_score = defaultdict(list)  # token mask rate in each done example
all_done_step_musk_token_num = defaultdict(list) # mask token num in each done example
all_done_step_unmusk_token_num = defaultdict(list)  # unmask token num in each done example


def batch_eval(eval_list):
    global game_step, input_special_token_ids, valid_ids_map

    eval_batch = len(eval_list) // config.eval_test_batch_size + 1

    all_result = None
    for i in range(eval_batch):
        if (i+1) * config.eval_test_batch_size > len(eval_list):
            small_batch = eval_list[i * config.eval_test_batch_size:]
        else:
            small_batch = eval_list[i * config.eval_test_batch_size:(i+1) * config.eval_test_batch_size]

        batch_size = len(small_batch)
        input_ids = torch.zeros((batch_size, input_special_token_ids.size(1)),dtype=torch.long)
        for j in range(batch_size):
            line = [int(token) for token in small_batch[j].split(" ")]
            masked_num = sum([1 for token_id in line if token_id == tokenizer.mask_token_id])
            
            for k in range(valid_token_num):
                input_ids[j, valid_ids_map[k]] = line[k]

            all_game_step_mask_rate[game_step].append(masked_num/valid_token_num)
            all_game_step_mask_token_num[game_step].append(masked_num)
            game_step += 1

        input_ids = input_ids + input_special_token_ids
        input_ids = send_to_device(input_ids, device)

        outputs = transformer_model(input_ids=input_ids, token_type_ids=input_token_type_ids)
        predict_proba = outputs.logits.detach().cpu()
        predict_proba = torch.softmax(predict_proba, dim=-1)
        if all_result is None:
            all_result = predict_proba
        else:
            all_result = torch.cat([all_result, predict_proba], dim=0)

    return all_result.numpy()


def check_result(input_ids):
    input_ids = torch.tensor([input_ids, ], dtype=torch.long)
    input_ids = input_ids.reshape(1, -1)
    input_ids = send_to_device(input_ids, device)
    outputs = transformer_model(input_ids, token_type_ids=input_token_type_ids)[0].detach().cpu()
    output_label = torch.argmax(outputs, dim=-1)[0].tolist()
    output_prob = torch.softmax(outputs, dim=-1)[0].tolist()
    return output_label, output_prob


explainer = LimeTextExplainer(class_names=label_names, bow=False, mask_string=str(tokenizer.mask_token_id))

for index, item in tqdm(enumerate(eval_dataset), total=len(eval_dataset), disable=config.disable_tqdm):

    all_eval_example_num += 1
    if isinstance(text_col_name, str):
        text = item[text_col_name]
        encode_res = tokenizer.encode_plus(text, add_special_tokens=True)
        input_ids = encode_res.input_ids
        input_token_type_ids = torch.tensor(encode_res.token_type_ids, dtype=torch.long, device=device)
    else:
        text = [item[text_col_name[0]], item[text_col_name[1]]]
        encode_res = tokenizer.encode_plus(text, add_special_tokens=True)
        input_ids = encode_res.input_ids
        input_token_type_ids = torch.tensor(encode_res.token_type_ids, dtype=torch.long, device=device)

    input_words = tokenizer.convert_ids_to_tokens(input_ids)

    golden_label = item["label"]
    raw_text = text
    original_pred_label, original_pred_prob = check_result(input_ids)
    original_pred_prob = original_pred_prob[original_pred_label]

    input_special_token_ids = torch.zeros((1,len(input_ids)), dtype=torch.long)
    valid_input_ids = []
    for i, token_id in enumerate(input_ids):
        if token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            valid_input_ids.append(token_id)
            valid_ids_map[valid_token_num] = i
            valid_token_num += 1
        else:
            input_special_token_ids[:,i] = token_id

    valid_input_ids = " ".join([str(i) for i in valid_input_ids])
    exp = explainer.explain_instance(valid_input_ids, batch_eval, labels=(original_pred_label,), 
                                     num_features=valid_token_num, num_samples=config.max_sample_num)

    exp_dict = exp.as_map()[original_pred_label]
    exp_dict = {i: value for i, value in exp_dict}

    musk_token_num = sum([1 for v in exp_dict.values() if v > 0])
    unmusk_token_num = valid_token_num - musk_token_num
    mask_rate = musk_token_num / valid_token_num

    attack_input_ids = input_ids.copy()
    fidelity_sub_input_ids = input_ids.copy()

    for i in range(valid_token_num):
        if exp_dict.get(i, -1) > 0:
            attack_input_ids[valid_ids_map[i]] = tokenizer.mask_token_id

    for i, token_id in enumerate(input_ids):
        if token_id not in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
            if attack_input_ids[i] != tokenizer.mask_token_id:
                fidelity_sub_input_ids[i] = tokenizer.mask_token_id

    attack_text = tokenizer.decode(attack_input_ids)

    all_musked_token_rate.append(mask_rate)
    all_unmusked_token_rate.append(1 - mask_rate)

    post_pred_label, post_pred_prob = check_result(attack_input_ids, )
    if post_pred_label != original_pred_label:
        attack_successful_num += 1
        fidelity_add.append(True)
        golden_label_name = label_names[golden_label]
        original_pred_label_name = label_names[original_pred_label]
        post_pred_label_name = label_names[post_pred_label]
        wandb_result_table.add_data(index, golden_label_name, original_pred_label_name, post_pred_label_name, raw_text, attack_text)
    else:
        fidelity_add.append(False)

    # delta prob
    post_pred_prob = post_pred_prob[original_pred_label]
    all_delta_prob.append(original_pred_prob - post_pred_prob)

    game_step = 0
    valid_token_num = 0
    valid_ids_map={}



reslut = {}
reslut["Eval Example Number"] = all_eval_example_num
reslut["Attack Success Rate"] = attack_successful_num / all_eval_example_num
reslut["Token Modification Rate"] = np.mean(all_musked_token_rate) # Token Level
reslut["Token Left Rate"] = np.mean(all_unmusked_token_rate) # Token Level
reslut["Fidelity"] = np.mean(fidelity_add)
reslut["delta_prob"] = np.mean(all_delta_prob)

logger.info(f"Result")
for k,v in reslut.items():
    logger.info(f"{k}: {v}")

if config.use_wandb:
    for k,v in reslut.items():
        wandb.run.summary[k] = v
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

