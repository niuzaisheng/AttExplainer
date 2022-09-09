
# Adversarial Attack results analysis process
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
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]

# the number of [CLS],[SEP] special token in an example
if isinstance(text_col_name, list):
    token_quantity_correction = 3
else:
    token_quantity_correction = 2

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer 
if config.features_type == "statistical_bin":
    input_feature_shape = 2 # 2D feature map
elif config.features_type == "const":
    input_feature_shape = 2 # 2D feature map
elif config.features_type == "random":
    input_feature_shape = 2 # 2D feature map
elif config.features_type == "effective_information":
    input_feature_shape = 1 # 1D feature map
elif config.features_type == "gradient":
    input_feature_shape = 2  # 2D feature map

if config.use_wandb:
    import wandb
    wandb.init(project=config.wandb_project_name, config=config)
    table_columns = ["completed_steps", "golden_label", "original_pred_label", "post_pred_label", "delta_p", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id


logger.info("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")


def get_rewards(original_seq_length, original_acc, original_prob, original_loss, post_acc, post_prob, post_loss, game_status, game_step):

    original_seq_length = (torch.FloatTensor(original_seq_length) - token_quantity_correction).to(game_status.device)
    unmusk_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmusked_token_rate = unmusk_token_num / original_seq_length
    musked_token_num = original_seq_length - unmusk_token_num
    musked_token_rate = 1 - unmusked_token_rate
    delta_p = (original_prob - post_prob)

    if config.task_type == "attack":
        if_success = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
        post_rewards = 10 * if_success

    elif config.task_type == "explain":
        if_success = (delta_p >= config.done_threshold).float()
        post_rewards = 10 * delta_p

    ifdone = if_success.clone() # die or win == 1
    for i in range(unmusk_token_num.size(0)):
        if unmusk_token_num[i] == 1:
            ifdone[i] = 1

    return post_rewards, ifdone, if_success, delta_p, musked_token_rate, unmusked_token_rate, musked_token_num, unmusk_token_num


def one_step(transformer_model, original_pred_labels, post_batch, seq_length, bins_num, lm_device, dqn_device, features_type):

    post_batch = send_to_device(post_batch, lm_device)
    if features_type != "gradient":
        with torch.no_grad():
            post_outputs = transformer_model(**post_batch, output_attentions=True)
            if features_type == "statistical_bin":
                extracted_features = get_attention_features(post_outputs, post_batch["attention_mask"], seq_length, bins_num)
            elif features_type == "const":
                extracted_features = get_const_attention_features(post_outputs, config.bins_num)
            elif features_type == "random":
                extracted_features = get_random_attention_features(post_outputs, config.bins_num)
            elif features_type == "effective_information":
                extracted_features = get_EI_attention_features(post_outputs, seq_length)

    else:
        post_outputs = transformer_model(**post_batch, output_attentions=True)
        extracted_features = get_gradient_features(post_outputs, seq_length, post_batch["input_ids"], original_pred_labels, embedding_weight_tensor)
        embedding_weight_tensor.grad.zero_()

    post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, device=dqn_device)
    post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, device=dqn_device)

    now_features = extracted_features.unsqueeze(1)

    return now_features, post_acc, post_loss, post_prob, post_pred_labels


dqn = DQN(config, do_eval=True , mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)
exp_name = "eval"
completed_steps = 0

# Metrics
attack_successful_num = 0  # Attack Success Rate
attack_example_num = 0  # Attack Success Rate
all_musked_token_rate = []  # Word Modification Rate
# all_musked_word_rate = []
all_unmusked_token_rate = []
all_success_game_step = []  # Average Victim Model Query Times
all_fidelity = []
all_delta_prob = []
all_cumulative_rewards = []

all_game_step_done_num = defaultdict(list)  # success rate in each game_step
all_game_step_mask_rate = defaultdict(list)  # token mask rate in each game_step
all_game_step_mask_token_num = defaultdict(list)  # mask token num in each game_step
all_done_step_musk_rate_score = defaultdict(list)  # token mask rate in each done example
all_done_step_musk_token_num = defaultdict(list)  # mask token num in each done example
all_done_step_unmusk_token_num = defaultdict(list)  # unmask token num in each done example
all_eval_token_length = []
all_eval_example_num = 0

transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

if config.features_type == "gradient":
    embedding_weight_tensor = transformer_model.get_input_embeddings().weight
    embedding_weight_tensor.requires_grad_(True)

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

    original_batch = clone_batch(batch)
    batch = send_to_device(batch, lm_device)
    empty_batch = get_empty_batch(batch, special_tokens_mask, mask_token_id=MASK_TOKEN_ID)

    with torch.no_grad():
        original_outputs = transformer_model(**batch, output_attentions=True)
        empty_outputs = transformer_model(**empty_batch, output_attentions=True)

    original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)
    original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

    batch_size = len(seq_length)
    batch_size_at_start = len(seq_length)
    all_eval_token_length.extend(seq_length)
    all_eval_example_num += len(seq_length)
    progress_bar.update(1)
    game_step_progress_bar.reset()
    attack_example_num += batch_size
    all_game_step_mask_rate[0].append(0)
    all_game_step_mask_token_num[0].append(0)
    cumulative_rewards = None

    original_seq_length = copy.deepcopy(seq_length)
    # initial batch
    actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, lm_device)
    now_features, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, batch, seq_length, config.bins_num,
                                                                                lm_device=lm_device, dqn_device=dqn.device, features_type = config.features_type)
    for game_step in range(epoch_game_steps):
        game_step_progress_bar.update()
        game_step_progress_bar.set_postfix({"left_examples": batch_size})
        all_game_step_done_num[game_step].append(1 - batch_size / batch_size_at_start)

        post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, special_tokens_mask, now_features, now_game_status)
        next_features, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config.bins_num,
                                                                                     lm_device=lm_device, dqn_device=dqn.device, features_type = config.features_type)
        rewards, ifdone, if_success, delta_p, musked_token_rate, unmusked_token_rate, \
            musked_token_num, unmusk_token_num = get_rewards(original_seq_length, original_acc, original_prob, original_loss,
                                                             post_acc, post_prob, post_loss, next_game_status, game_step)

        if cumulative_rewards is None:
            cumulative_rewards = rewards
        else:
            cumulative_rewards += rewards

        all_game_step_mask_rate[game_step + 1].extend(musked_token_rate.tolist())
        all_game_step_mask_token_num[game_step + 1].extend(musked_token_num.tolist())

        removed_index = [i for i in range(batch_size) if ifdone[i].item() == 1]
        if len(removed_index) != 0:
            success_index = [i for i in range(batch_size) if if_success[i].item() == 1]
            all_musked_token_rate.extend(musked_token_rate[removed_index].tolist())
            all_unmusked_token_rate.extend(unmusked_token_rate[removed_index].tolist())
            all_success_game_step.extend([game_step + 1 ] * len(removed_index))
            attack_successful_num += len(success_index)
            all_cumulative_rewards.extend(cumulative_rewards[removed_index].tolist())
            all_done_step_musk_rate_score[game_step + 1].extend(musked_token_rate[removed_index].tolist())
            all_done_step_musk_token_num[game_step + 1].extend(musked_token_num[removed_index].tolist())
            all_done_step_unmusk_token_num[game_step + 1].extend(unmusk_token_num[removed_index].tolist())

            done_seq_length = [v for i, v in enumerate(original_seq_length) if i in removed_index]
            # done_seq_length = original_seq_length[removed_index]
            done_token_word_position_map = [v for i, v in enumerate(token_word_position_map) if i in removed_index]
            # word_masked_rate = get_word_masked_rate(now_game_status[removed_index], done_seq_length, done_token_word_position_map)
            # all_musked_word_rate.extend(word_masked_rate)

            # fidelity
            finished_index = removed_index
            if config.token_replacement_strategy == "mask":
                fidelity_acc, _ = compute_fidelity_when_masked(transformer_model, finished_index, post_batch,
                                                special_tokens_mask, now_game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)
            elif config.token_replacement_strategy == "delete":
                fidelity_acc = compute_fidelity_when_deleted(transformer_model, finished_index, post_batch,
                                                special_tokens_mask, now_game_status, original_pred_labels, lm_device)

            all_fidelity.extend(fidelity_acc.tolist())

            # delta_prob
            all_delta_prob.extend(delta_p[removed_index].tolist())

            if config.use_wandb:
                save_result(len(removed_index), completed_steps, original_batch["input_ids"][removed_index], post_batch["input_ids"][removed_index],
                            golden_labels[removed_index], original_pred_labels[removed_index], post_pred_labels[removed_index],
                            delta_p[removed_index], tokenizer, label_names, wandb_result_table)

        # Remove those completed samples and parpare for next game step
        batch_size, seq_length, original_seq_length, golden_labels, special_tokens_mask, now_features, \
                now_game_status, original_batch, batch, original_pred_labels, \
                token_word_position_map, cumulative_rewards, \
                original_acc, original_loss, original_prob = \
                gather_unfinished_examples(ifdone, batch_size, seq_length, original_seq_length, golden_labels, next_special_tokens_mask,
                                           next_features, next_game_status,
                                           original_batch, post_batch, original_pred_labels,
                                           token_word_position_map, cumulative_rewards,
                                           original_acc, original_loss, original_prob)
    
        if config.token_replacement_strategy == "delete":
            seq_length = [ x-1 for x in seq_length]

        if batch_size == 0:
            if game_step + 2 <= epoch_game_steps:
                for future_step in range(game_step + 2, epoch_game_steps + 1):
                    all_game_step_done_num[future_step].append(1)
            break

        completed_steps += 1

    if batch_size != 0:
        # Not successful samples
        unfinished_index = [i for i in range(batch_size)]

        all_delta_prob.extend(delta_p.tolist())
        all_musked_token_rate.extend(musked_token_rate.tolist())
        all_unmusked_token_rate.extend(unmusked_token_rate.tolist())

        # word_masked_rate = get_word_masked_rate(now_game_status, original_seq_length, token_word_position_map)
        # all_musked_word_rate.extend(word_masked_rate)

        fidelity_acc, _ = compute_fidelity_when_masked(transformer_model, unfinished_index, post_batch,
                                           special_tokens_mask, next_game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)
        all_fidelity.extend(fidelity_acc.tolist())

        if config.use_wandb:
            save_result(len(unfinished_index), completed_steps, original_batch["input_ids"], post_batch["input_ids"],
                        golden_labels, original_pred_labels, post_pred_labels, delta_p, tokenizer, label_names, wandb_result_table)

    all_game_step_done_num[game_step + 1].append(1 - batch_size / batch_size_at_start)

try:
    assert attack_example_num == len(all_delta_prob) == len(all_fidelity) == \
        len(all_musked_token_rate) == len(all_unmusked_token_rate) # == len(all_musked_word_rate)
except:
    print(f"""{attack_example_num} == {len(all_delta_prob)} == {len(all_fidelity)} ==
          {len(all_musked_token_rate)} == {len(all_unmusked_token_rate)}""")


logger.info("Finish eval!")

reslut = {}
reslut["Eval Example Number"] = all_eval_example_num
reslut["Average Eval Token Length"] = np.mean(all_eval_token_length)
reslut["Attack Success Rate"] = attack_successful_num / attack_example_num
reslut["Token Modification Rate"] = np.mean(all_musked_token_rate)
# reslut["Word Modification Rate"] = np.mean(all_musked_word_rate)
reslut["Word Left Rate"] = np.mean(all_unmusked_token_rate)
reslut["Average Victim Model Query Times"] = np.mean(all_success_game_step)
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
