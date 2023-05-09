
# %%
import argparse
import copy
import logging
import sys
from collections import defaultdict

import numpy as np
import torch
from accelerate.utils import send_to_device
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data_utils import get_dataloader_and_model, get_dataset_config
from dqn_model import DQN
from utils import *

logger = logging.getLogger(__name__)

set_seed(43)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack analysis process")

    parser.add_argument(
        "--data_set_name", type=str, default="esnli", help="The name of the dataset. On of emotion,snli or sst2."
    )
    # parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])
    parser.add_argument("--gpu_index", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner-dev")
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--discribe", type=str, default="Model evaluation process")

    args = parser.parse_args()
    return args

config = parse_args()
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO
)
logger.info(f"Eval config: {config}")
set_seed(config.seed)

gpu_index = config.gpu_index if config.gpu_index is not None else get_most_free_gpu_index()
lm_device = torch.device("cuda", gpu_index)
process_device = torch.device("cpu")

dataset_config = get_dataset_config(config)
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]

# the number of [CLS],[SEP] special token in an example
if isinstance(text_col_name, list):
    token_quantity_correction = 3
else:
    token_quantity_correction = 2

# if config.use_wandb:
#     import wandb
#     wandb.init(project=config.wandb_project_name, config=config)
#     table_columns = ["completed_steps", "golden_label", "original_pred_label", "post_pred_label", "delta_p", "original_input_ids", "post_batch_input_ids"]
#     wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

# %%
logger.info("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")

transformer_model.to(lm_device)
transformer_model.eval()


def one_step(transformer_model, original_pred_labels, post_batch, seq_length, lm_device, process_device):
    post_batch = send_to_device(post_batch, lm_device)
    with torch.no_grad():
        post_outputs = transformer_model(**post_batch, output_attentions=True)

    post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, process_device)
    post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, process_device)

    return post_acc, post_loss, post_prob, post_pred_labels

def mask_token_by_evidence(batch, MASK_TOKEN_ID):
    batch = clone_batch(batch)
    evidence_token_mask = batch["evidence_token_mask"]
    batch["input_ids"] = batch["input_ids"].masked_fill(evidence_token_mask, MASK_TOKEN_ID)
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "token_type_ids": batch["token_type_ids"]
    }

def delete_token_by_evidence(batch):
    # evidence_token_mask is same shape as input_ids
    # mask is 1 where token is evidence token
    # mask is 0 where token is not evidence token
    batch_size, last_max_seq_len = batch["input_ids"].shape
    target_device = batch["input_ids"].device
    evidence_token_mask = batch["evidence_token_mask"].bool()

    post_input_ids = torch.zeros((batch_size, last_max_seq_len), dtype=torch.long, device=target_device)
    post_attention_mask = torch.zeros((batch_size, last_max_seq_len), dtype=torch.long, device=target_device)
    post_token_type_ids = torch.zeros((batch_size, last_max_seq_len), dtype=torch.long, device=target_device)
    special_tokens_mask = torch.zeros((batch_size, last_max_seq_len), dtype=torch.bool, device=target_device)

    for i in range(batch_size):
        example_input_ids =  batch["input_ids"][i].masked_select(~evidence_token_mask[i])
        post_input_ids[i, :example_input_ids.shape[0]] = example_input_ids
        post_attention_mask[i, :example_input_ids.shape[0]] = 1
        post_token_type_ids[i, :example_input_ids.shape[0]] = batch["token_type_ids"][i, :example_input_ids.shape[0]]
        special_tokens_mask[i, :example_input_ids.shape[0]] = batch["special_tokens_mask"][i, :example_input_ids.shape[0]]

    return {
        "input_ids": post_input_ids,
        "attention_mask": post_attention_mask,
        "token_type_ids": post_token_type_ids,
    }


# %%
all_acc = defaultdict(list)
all_loss = defaultdict(list)
all_prob = defaultdict(list)
all_pred_labels = defaultdict(list)

for batch in tqdm(eval_dataloader, disable=config.disable_tqdm):

    if batch.get("id") is not None:
        ids = batch.pop("id")
    seq_length = batch.pop("seq_length")

    batch_max_seq_length = max(seq_length)
    golden_labels = batch.pop("labels")
    token_word_position_map = batch.pop("token_word_position_map")

    deleted_batch = delete_token_by_evidence(batch)
    masked_batch = mask_token_by_evidence(batch, MASK_TOKEN_ID)
    special_tokens_mask = batch.pop("special_tokens_mask")
    evidence_token_mask = batch.pop("evidence_token_mask")

    original_acc, original_loss, original_prob, original_pred_labels = one_step(transformer_model, golden_labels, batch, seq_length, lm_device, process_device)
    correctly_classified_index_mask = (original_acc == 1).bool()
    all_acc["original"].append(original_acc)
    all_loss["original"].append(original_loss)
    all_prob["original"].append(original_prob)
    all_pred_labels["original"].append(original_pred_labels)

    deleted_acc, deleted_loss, deleted_prob, deleted_pred_labels = one_step(transformer_model, golden_labels, deleted_batch, seq_length, lm_device, process_device)
    all_acc["deleted"].append(deleted_acc)
    all_loss["deleted"].append(deleted_loss)
    all_prob["deleted"].append(deleted_prob)
    all_pred_labels["deleted"].append(deleted_pred_labels)

    all_acc["correctly_classified_deleted"].append(deleted_acc[correctly_classified_index_mask])
    all_loss["correctly_classified_deleted"].append(deleted_loss[correctly_classified_index_mask])
    all_prob["correctly_classified_deleted"].append(deleted_prob[correctly_classified_index_mask])
    all_pred_labels["correctly_classified_deleted"].append(deleted_pred_labels[correctly_classified_index_mask])

    masked_acc, masked_loss, masked_prob, masked_pred_labels = one_step(transformer_model, golden_labels, masked_batch, seq_length, lm_device, process_device)
    all_acc["masked"].append(masked_acc)
    all_loss["masked"].append(masked_loss)
    all_prob["masked"].append(masked_prob)
    all_pred_labels["masked"].append(masked_pred_labels)

    all_acc["correctly_classified_masked"].append(masked_acc[correctly_classified_index_mask])
    all_loss["correctly_classified_masked"].append(masked_loss[correctly_classified_index_mask])
    all_prob["correctly_classified_masked"].append(masked_prob[correctly_classified_index_mask])
    all_pred_labels["correctly_classified_masked"].append(masked_pred_labels[correctly_classified_index_mask])

# torch concat all list
for key in all_acc.keys():
    all_acc[key] = torch.cat(all_acc[key])
    all_loss[key] = torch.cat(all_loss[key])
    all_prob[key] = torch.cat(all_prob[key])
    all_pred_labels[key] = torch.cat(all_pred_labels[key])

# print result
for key in all_acc.keys():
    
    print(f"{key} acc: {all_acc[key].mean():.2f}")
    print(f"{key} loss: {all_loss[key].mean():.2f}")
    print(f"{key} avg prob: {all_prob[key].mean():.2f}")


