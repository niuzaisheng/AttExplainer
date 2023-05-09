# Model interpretability training process
import argparse
import copy
import datetime
import logging
import os
import sys

import numpy as np
import torch
from accelerate.utils import send_to_device
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data_utils import get_dataloader_and_model, get_dataset_config
from dqn_model import DQN
from utils import *

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN training process")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )

    parser.add_argument("--task_type", type=str, default="attack",
                        choices=["attack", "explain"],
                        help="The type of the task. On of attack or explain.")

    # Dqn Settings
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--max_memory_capacity", type=int, default=10000)
    parser.add_argument("--dqn_rl", type=float, default=0.0001)
    parser.add_argument("--target_replace_iter", type=int, default=100)
    parser.add_argument("--dqn_batch_size", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--use_ddqn", action="store_true", default=True)
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    # Game Settings
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["const", "random", "input_ids", "original_embedding",
                                 "statistical_bin", "effective_information",
                                 "gradient", "gradient_input", "mixture"])
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--do_pre_deletion", action="store_true", default=False, help="Pre-deletion of misclassified samples")
    parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])

    # Train settings
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--is_agent_on_GPU", type=bool, default=True)
    parser.add_argument("--max_train_epoch", type=int, default=1000)
    parser.add_argument("--max_sampling_steps", type=int, default=None)
    parser.add_argument("--save_step_iter", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner-dev")
    parser.add_argument("--disable_tqdm", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discribe", type=str, default="DQN model training process")

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

set_seed(config.seed)

dt = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
exp_name = f"{config.data_set_name}_{dt}"
save_file_dir = f"saved_weights/{exp_name}"

dataset_config = get_dataset_config(config)
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]
token_quantity_correction = dataset_config["token_quantity_correction"]  # the number of [CLS],[SEP] special token in an example

# For DQN gather single step data into batch from replay buffer
# input_feature_shape indicates how many dimensions along sequence length
input_feature_shape = input_feature_shape_dict[config.features_type]


if config.use_wandb:
    import wandb
    wandb.init(name=exp_name, project=config.wandb_project_name, config=config)
    wandb_result_table = create_result_table("train")

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id
config.vocab_size = tokenizer.vocab_size

logger.info("Start loading data and model!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")

logger.info("One example:")
for _, batch in enumerate(simulate_dataloader):
    logger.info(batch)
    break


def record_results(completed_steps, transformer_model, trackers, finished_index, post_batch,
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
            trackers[batch_index].save_result_row(tokenizer, label_names, wandb_result_table, completed_steps=completed_steps)


def get_rewards(original_seq_length=None,
                original_acc=None, original_prob=None, original_logits=None, original_loss=None,
                post_acc=None, post_prob=None, post_logits=None, post_loss=None,
                game_status=None, game_step=None):

    device = game_status.device
    original_seq_length = (torch.FloatTensor(original_seq_length).to(device) - token_quantity_correction)
    unmask_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmasked_token_rate = unmask_token_num / original_seq_length
    masked_token_num = original_seq_length - unmask_token_num
    masked_token_rate = 1 - unmasked_token_rate
    delta_prob = (original_prob - post_prob)
    delta_logits = None
    if original_logits is not None and post_logits is not None:
        delta_logits = (original_logits - post_logits)
    delta_loss = None
    if original_loss is not None and post_loss is not None:
        delta_loss = (original_loss - post_loss)

    if config.task_type == "attack":
        if_success = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
        rewards = delta_prob + 10 * if_success * unmasked_token_rate # default reward setting
        # rewards = delta_prob + 10 * if_success # ablation study
        # rewards = 10 * if_success * unmasked_token_rate # ablation study
        # rewards = delta_prob # ablation study

    elif config.task_type == "explain":
        if_success = (delta_prob >= config.done_threshold).float()
        rewards = delta_prob + 10 * if_success * unmasked_token_rate # default reward setting
        # rewards = delta_prob + 10 * if_success # ablation study
        # rewards = 10 * if_success * unmasked_token_rate # ablation study
        # rewards = delta_prob # ablation study

    if_done = if_success.clone()  # die or win == 1
    for i in range(unmask_token_num.size(0)):
        if unmask_token_num[i] == 0:  # end of game
            if_done[i] = 1

    return GameEnvironmentVariables(
        rewards=rewards,
        if_done=if_done,
        if_success=if_success,
        delta_prob=delta_prob,
        masked_token_rate=masked_token_rate,
        unmasked_token_rate=unmasked_token_rate,
        masked_token_num=masked_token_num,
        unmask_token_num=unmask_token_num,
        delta_logits=delta_logits,
        delta_loss=delta_loss,
    )


def one_step(transformer_model, original_pred_labels, post_batch, seq_length, bins_num, lm_device, dqn_device, features_type):

    post_batch = send_to_device(post_batch, lm_device)
    if features_type == "gradient":
        extracted_features, post_outputs = get_gradient_features(transformer_model, post_batch, original_pred_labels, times_input=False)
    elif features_type == "gradient_input":
        extracted_features, post_outputs = get_gradient_features(transformer_model, post_batch, original_pred_labels, times_input=True)
    elif features_type == "original_embedding":
        extracted_features, post_outputs = use_original_embedding_as_features(transformer_model, post_batch)
    elif features_type == "mixture":
        extracted_features, post_outputs = get_mixture_features(transformer_model, post_batch, original_pred_labels, seq_length, bins_num)
    else:
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
            elif features_type == "input_ids":
                extracted_features = post_batch["input_ids"]
            else:
                raise NotImplementedError(f"features_type {features_type} not implemented")

    post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, device=dqn_device)
    # post_logits = batch_logits(post_outputs, original_pred_labels, device=dqn_device)
    # post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, device=dqn_device)
    now_features = extracted_features.unsqueeze(1)

    return now_features, post_acc, post_prob, post_pred_labels


dqn = DQN(config, do_eval=False, mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)
progress_bar = tqdm(total=config.max_train_epoch * len(simulate_dataloader), disable=config.disable_tqdm)
exp_name = "train"

status_dict = {}  # for tqdm display some info

def update_dict(any_dict, step):
    status_dict.update(any_dict)
    progress_bar.set_postfix(status_dict)
    if config.use_wandb:
        wandb.log({exp_name: any_dict}, step=step)


lm_device = torch.device("cuda", config.gpu_index)
transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

completed_steps = 0
if_achieve_max_sampling_steps = False

for epoch in range(config.max_train_epoch):
    update_dict({"epoch": epoch}, completed_steps)

    for step, batch in enumerate(simulate_dataloader):

        ids = batch.pop("id")
        seq_length = batch.pop("seq_length")  # vailid tokens num in each sequence, a list of int
        batch_max_seq_length = max(seq_length)
        golden_labels = batch.pop("labels")
        special_tokens_mask = batch.pop("special_tokens_mask").bool()
        special_tokens_mask = send_to_device(special_tokens_mask, dqn.device)
        token_word_position_map = batch.pop("token_word_position_map")
        original_batch = clone_batch(batch)
        batch = send_to_device(batch, lm_device)

        with torch.no_grad():
            original_outputs = transformer_model(**batch, output_attentions=True)

        original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)
        original_logits = batch_logits(original_outputs, original_pred_labels, device=dqn.device)
        original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

        # update_dict({"original_prob": original_prob.mean().item()}, completed_steps)
        trackers = create_trackers(ids, seq_length, batch["input_ids"], token_word_position_map,
                                   golden_labels, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                                   token_quantity_correction, config.token_replacement_strategy)

        batch_size = len(seq_length)
        if config.do_pre_deletion:
            batch_size, seq_length, special_tokens_mask, batch, \
                original_loss, original_acc, original_pred_labels, original_prob = \
                gather_correct_examples(original_acc, batch_size, seq_length, special_tokens_mask, batch,
                                        original_loss, original_pred_labels, original_prob)
        if batch_size == 0:
            continue

        batch_size = len(seq_length)
        batch_size_at_start = len(seq_length)
        original_seq_length = copy.deepcopy(seq_length)

        actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, dqn.device)
        now_features, post_acc, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, batch, seq_length, config.bins_num,
                                                                       lm_device=lm_device, dqn_device=dqn.device, features_type=config.features_type)

        batch_done_step = []

        for game_step in range(config.max_game_steps):

            post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, special_tokens_mask, now_features, now_game_status)
            next_features, post_acc, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config.bins_num,
                                                                            lm_device=lm_device, dqn_device=dqn.device, features_type=config.features_type)

            r = get_rewards(original_seq_length=original_seq_length, original_acc=original_acc, original_prob=original_prob, post_acc=post_acc, post_prob=post_prob, game_status=next_game_status)

            dqn.store_transition(batch_size, special_tokens_mask, next_special_tokens_mask, now_features, next_features, now_game_status, next_game_status, actions, seq_length, r.rewards, r.if_success)

            update_trackers(trackers, variables=r, action=actions, post_prob=post_prob, post_pred_label=post_pred_labels.view(-1))

            success_index = [i for i in range(batch_size) if r.if_done[i].item() == 1]
            if len(success_index) != 0:
                batch_done_step.extend([game_step] * len(success_index))
                if completed_steps % 2 == 0:
                    finished_index = success_index
                    update_dict({
                        "done_rewards": r.rewards[success_index].mean().item(),
                        "done_masked_token_rate": r.masked_token_rate[success_index].mean().item(),
                        "done_unmasked_token_rate": r.unmasked_token_rate[success_index].mean().item(),
                        "done_delta_prob": r.delta_prob[success_index].mean().item(),
                    }, step=completed_steps)
                # if completed_steps % 100 == 0:
                #     record_results(completed_steps, transformer_model, trackers, success_index, post_batch, next_special_tokens_mask, next_game_status, original_pred_labels, lm_device)

            if completed_steps % 10 == 0:
                update_dict({
                    "post_acc": post_acc.mean().item(),
                    "rewards": r.rewards.mean().item(),
                    "unmasked_token_rate": r.unmasked_token_rate.mean().item(),
                    "masked_token_rate": r.masked_token_rate.mean().item(),
                    "if_done": r.if_done.mean().item(),
                    "delta_prob": r.delta_prob.mean().item(),
                }, step=completed_steps)

            batch_size, trackers, seq_length, original_seq_length, original_acc, original_pred_labels, original_prob, original_logits, original_loss, \
                special_tokens_mask, now_features, now_game_status, batch = \
                gather_unfinished_examples_with_tracker(r.if_done, trackers, seq_length,
                                                        original_seq_length, original_acc, original_pred_labels, original_prob, original_logits, original_loss,
                                                        next_special_tokens_mask, next_features, next_game_status, post_batch)

            if config.token_replacement_strategy == "delete":
                seq_length = [x-1 for x in seq_length]

            completed_steps += 1
            if config.max_sampling_steps is not None and completed_steps >= config.max_sampling_steps:
                if_achieve_max_sampling_steps = True
                logger.info("achieve max sampling steps")
                break
            if completed_steps % 10 == 9:
                dqn_loss = dqn.learn()
                update_dict({"dqn_loss": dqn_loss}, step=completed_steps)

            if completed_steps % config.save_step_iter == 0:
                if not os.path.isdir(save_file_dir):
                    os.mkdir(save_file_dir)
                file_name = f"{save_file_dir}/dqn-{completed_steps}.bin"
                with open(file_name, "wb") as f:
                    torch.save(dqn.eval_net.state_dict(), f)
                    logger.info(f"checkpoint saved in {file_name}")

                # BUG record_results original_pred_labels is reflash by gather_unfinished_examples_with_tracker
                # record_results(completed_steps, transformer_model, trackers, success_index, post_batch, next_special_tokens_mask, next_game_status, original_pred_labels, lm_device)

            if batch_size == 0:
                break

        done_rate = 1 - batch_size / batch_size_at_start
        update_dict({"average_done_step": np.mean(batch_done_step), "done_rate": done_rate}, step=completed_steps)
        progress_bar.update(1)
        if if_achieve_max_sampling_steps: break

    if if_achieve_max_sampling_steps: break

logger.info("Finish training!")
if config.use_wandb:
    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()

with open(f"{save_file_dir}/dqn-final.bin", "wb") as f:
    torch.save(dqn.eval_net.state_dict(), f)
logger.info(f"Finish saving in {save_file_dir}")
