
import argparse
import copy
import logging
import sys
from collections import defaultdict, Counter

import numpy as np
import torch
from accelerate.utils import send_to_device
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data_utils import (get_dataloader_and_model, get_dataset_config,
                        get_word_masked_rate)
from dqn_model import DQN
from utils import *

logger = logging.getLogger(__name__)

set_seed(43)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack analysis process")

    parser.add_argument("--task_type", type=str, default="attack",
                        choices=["attack", "explain"],
                        help="The type of the task. On of attack or explain.")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["statistical_bin", "const", "random", "effective_information", "gradient"],)
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])
    parser.add_argument("--use_ddqn", action="store_true", default=False)
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--is_agent_on_GPU", type=bool, default=False)
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
lm_device = torch.device("cuda", config.gpu_index)

dataset_config = get_dataset_config(config)
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
token_quantity_correction = dataset_config["token_quantity_correction"]  # the number of [CLS],[SEP] special token in an example

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape = input_feature_shape_dict[config.features_type]

if config.use_wandb:
    import wandb
    wandb.init(project=config.wandb_project_name, config=config)
    wandb_result_table = create_result_table()


tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id


logger.info("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")


def get_rewards(original_seq_length,
                original_acc, original_prob, original_logits, original_loss,
                post_acc, post_prob, post_logits, post_loss,
                game_status, game_step):

    original_seq_length = (torch.FloatTensor(original_seq_length) - token_quantity_correction).to(game_status.device)
    unmask_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmasked_token_rate = unmask_token_num / original_seq_length
    masked_token_num = original_seq_length - unmask_token_num
    masked_token_rate = 1 - unmasked_token_rate
    delta_prob = (original_prob - post_prob)
    delta_logits = (original_logits - post_logits)

    if config.task_type == "attack":
        if_success = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
        post_rewards = 10 * if_success

    elif config.task_type == "explain":
        if_success = (delta_prob >= config.done_threshold).float()
        post_rewards = 10 * delta_prob

    if_done = if_success.clone()  # die or win == 1
    for i in range(unmask_token_num.size(0)):
        if unmask_token_num[i] == 1:
            if_done[i] = 1

    return post_rewards, if_done, if_success, delta_prob, delta_logits, \
        masked_token_rate, unmasked_token_rate, masked_token_num, unmask_token_num


def one_step(transformer_model, original_pred_labels, post_batch, seq_length, config, lm_device, dqn_device):

    features_type = config.features_type
    post_batch = send_to_device(post_batch, lm_device)
    if features_type != "gradient":
        with torch.no_grad():
            post_outputs = transformer_model(**post_batch, output_attentions=True)
            if features_type == "statistical_bin":
                extracted_features = get_attention_features(post_outputs, post_batch["attention_mask"], seq_length, config.bins_num)
            elif features_type == "const":
                extracted_features = get_const_attention_features(post_outputs, config.bins_num)
            elif features_type == "random":
                extracted_features = get_random_attention_features(post_outputs, config.bins_num)
            elif features_type == "effective_information":
                extracted_features = get_EI_attention_features(post_outputs, seq_length)

    else:
        extracted_features, post_outputs = get_gradient_features(transformer_model, post_batch, original_pred_labels)

    post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, device=dqn_device)
    post_logits = batch_logits(post_outputs, original_pred_labels, device=dqn.device)
    post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, device=dqn_device)

    now_features = extracted_features.unsqueeze(1)

    return now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss


def record_results(transformer_model, trackers, finished_index, post_batch,
                   special_tokens_mask, game_status, original_pred_labels, lm_device):
    if config.token_replacement_strategy == "mask":
        fidelity_acc, _ = compute_fidelity_when_masked(transformer_model, finished_index, post_batch,
                                                       special_tokens_mask, game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)
    elif config.token_replacement_strategy == "delete":
        fidelity_acc = compute_fidelity_when_deleted(transformer_model, finished_index, post_batch,
                                                     special_tokens_mask, game_status, original_pred_labels, lm_device)

    for i, batch_index in enumerate(finished_index):
        trackers[batch_index].fidelity = fidelity_acc[i].item()
        trackers[batch_index].post_input_ids = post_batch["input_ids"][batch_index]
        if config.use_wandb:
            trackers[batch_index].save_result_row(tokenizer, label_names, wandb_result_table)


dqn = DQN(config, do_eval=True, mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)
exp_name = "eval"

# Metrics and stage tracker
all_trackers = []

transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

epoch_game_steps = config.max_game_steps
progress_bar = tqdm(desc="example", total=len(eval_dataloader), disable=config.disable_tqdm)
game_step_progress_bar = tqdm(desc="game_step", total=epoch_game_steps, disable=config.disable_tqdm)

for step, batch in enumerate(eval_dataloader):

    seq_length = batch.pop("seq_length")
    if batch.get("id") is not None:
        ids = batch.pop("id")
    batch_max_seq_length = max(seq_length)
    golden_labels = batch.pop("labels")
    special_tokens_mask = batch.pop("special_tokens_mask")
    special_tokens_mask = send_to_device(special_tokens_mask, dqn.device)
    token_word_position_map = batch.pop("token_word_position_map")

    batch = send_to_device(batch, lm_device)
    # empty_batch = get_empty_batch(batch, special_tokens_mask, mask_token_id=MASK_TOKEN_ID)

    with torch.no_grad():
        original_outputs = transformer_model(**batch)
        # empty_outputs = transformer_model(**empty_batch)

    original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)
    original_logits = batch_logits(original_outputs, original_pred_labels, device=dqn.device)
    original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

    batch_size = len(seq_length)
    batch_size_at_start = len(seq_length)

    progress_bar.update(1)
    game_step_progress_bar.reset()
    trackers = create_trackers(ids, seq_length, batch["input_ids"], token_word_position_map,
                               golden_labels, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                               token_quantity_correction, config.token_replacement_strategy)
    all_trackers.extend(trackers)

    original_seq_length = copy.deepcopy(seq_length)
    # initial batch
    actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, lm_device)
    now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, batch, seq_length, config,
                                                                                           lm_device=lm_device, dqn_device=dqn.device)
    for game_step in range(epoch_game_steps):
        game_step_progress_bar.update()
        game_step_progress_bar.set_postfix({"left_examples": batch_size})

        post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, special_tokens_mask, now_features, now_game_status)
        next_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config,
                                                                                                lm_device=lm_device, dqn_device=dqn.device)
        rewards, if_done, if_success, delta_prob, delta_logits, masked_token_rate, unmasked_token_rate, \
            masked_token_num, unmask_token_num = get_rewards(original_seq_length,
                                                             original_acc, original_prob, original_logits, original_loss,
                                                             post_acc, post_prob, post_logits, post_loss,
                                                             next_game_status, game_step)

        update_trackers(trackers, action=actions, if_done=if_done, if_success=if_success, reward=rewards,
                        post_prob=post_prob, post_logits=post_logits, post_loss=post_loss,
                        delta_prob=delta_prob, delta_logits=delta_logits,
                        masked_token_rate=masked_token_rate, masked_token_num=masked_token_num)

        success_index = [i for i in range(batch_size) if if_done[i].item() == 1]
        if len(success_index) != 0:
            record_results(transformer_model, trackers, success_index, post_batch, special_tokens_mask, next_game_status, original_pred_labels, lm_device)

        # Remove those completed samples and parpare for next game step
        batch_size, trackers, seq_length, original_seq_length, original_acc, original_pred_labels, original_prob, original_logits, original_loss, special_tokens_mask, now_features, now_game_status, batch = \
            gather_unfinished_examples_with_tracker(if_done, trackers, seq_length,
                                                    original_seq_length, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                                                    special_tokens_mask, next_features, next_game_status, batch)
        if batch_size == 0:
            break
        if config.token_replacement_strategy == "delete":
            seq_length = [x-1 for x in seq_length]

    if batch_size != 0:
        # Not successful samples
        fail_index = [i for i in range(batch_size)]
        for i in fail_index:
            trackers[i].done_and_fail()
        record_results(transformer_model, trackers, fail_index, post_batch, special_tokens_mask, next_game_status, original_pred_labels, lm_device)

logger.info("Finish eval!")
logger.info("Saving results...")

reslut = {}
reslut["Eval Example Number"] = len(all_trackers)
reslut["Average Eval Token Length"] = np.mean([item.original_seq_length for item in all_trackers])
reslut["Attack Success Rate"] = sum([1 for item in all_trackers if item.if_success]) / len(all_trackers)
reslut["Token Modification Rate"] = np.mean([item.masked_token_rate[-1] for item in all_trackers])
reslut["Token Modification Number"] = np.mean([item.masked_token_num[-1] for item in all_trackers])
# reslut["Word Modification Rate"] = np.mean(all_masked_word_rate)
# reslut["Word Left Rate"] = np.mean(all_unmasked_token_rate)
reslut["Average Victim Model Query Times"] = np.mean([item.done_step for item in all_trackers])
reslut["Fidelity"] = np.mean([item.fidelity for item in all_trackers])
reslut["delta_prob"] = np.mean([item.delta_prob[-1] for item in all_trackers])

logger.info(f"Result")
for k, v in reslut.items():
    logger.info(f"{k}: {v}")

if config.use_wandb:
    for k, v in reslut.items():
        wandb.run.summary[k] = v

    data = [[k, v] for k, v in Counter([item.done_step for item in all_trackers]).items()]
    table = wandb.Table(data=data, columns=["game_step", "attack_success_num"])
    wandb.log({"trade_off": wandb.plot.histogram(table, "game_step", "attack_success_num", title="Attack Success Num & Game Step Trade Off")})

    all_game_step_mask_rate = defaultdict(list)
    all_done_step_mask_rate = defaultdict(list)
    all_game_step_mask_token_num = defaultdict(list)
    all_done_step_mask_token_num = defaultdict(list)
    all_game_step_delta_prob = defaultdict(list)
    all_done_step_delta_prob = defaultdict(list)

    for item in all_trackers:
        # track all game_step
        for i in range(len(item.masked_token_rate)):
            all_game_step_mask_rate[i].append(item.masked_token_rate[i])
            all_game_step_mask_token_num[i].append(item.masked_token_num[i])
            all_game_step_delta_prob[i].append(item.delta_prob[i])

        # only track the last done_step
        all_done_step_mask_rate[item.done_step].append(item.masked_token_rate[-1])
        all_done_step_mask_token_num[item.done_step].append(item.masked_token_num[-1])
        all_done_step_delta_prob[item.done_step].append(item.delta_prob[-1])

    data = [[k, np.mean(v)] for k, v in all_game_step_mask_rate.items()]
    table2 = wandb.Table(data=data, columns=["game_step", "masked_rate"])
    wandb.log({"trade_off2": wandb.plot.histogram(table2, "game_step", "masked_rate", title="Musked Rate & Game Step Trade Off")})

    data = [[k, np.mean(v)] for k, v in all_done_step_mask_rate.items()]
    table3 = wandb.Table(data=data, columns=["done_step", "masked_rate"])
    wandb.log({"trade_off3": wandb.plot.scatter(table3, "done_step", "masked_rate", title="Musked Rate & Done Step Trade Off")})

    data = [[k, np.mean(v)] for k, v in all_game_step_mask_token_num.items()]
    table4 = wandb.Table(data=data, columns=["game_step", "mask_token_num"])
    wandb.log({"trade_off4": wandb.plot.line(table4, "game_step", "mask_token_num", title="Musked Token Number & Game Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_done_step_mask_token_num.items()]
    table5 = wandb.Table(data=data, columns=["done_step", "mask_token_num"])
    wandb.log({"trade_off5": wandb.plot.scatter(table5, "done_step", "mask_token_num", title="Musked Token Number & Done Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_game_step_delta_prob.items()]
    table6 = wandb.Table(data=data, columns=["game_step", "delta_prob"])
    wandb.log({"trade_off6": wandb.plot.scatter(table6, "game_step", "delta_prob", title="Delta Prob & Game Step Trade Off")})

    data = [[step, np.mean(value_list)] for (step, value_list) in all_done_step_delta_prob.items()]
    table7 = wandb.Table(data=data, columns=["done_step", "delta_prob"])
    wandb.log({"trade_off7": wandb.plot.scatter(table7, "done_step", "delta_prob", title="Delta Prob & Done Step Trade Off")})

    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()
