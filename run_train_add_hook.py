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
from language_model import MyBertForSequenceClassification, new_forward_func
from tasks import *
from utils import *


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DQN training process")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )

    parser.add_argument("--task_type", type=str, default="AttentionScoreAttackTask",
                        choices=["AttentionScoreAttackTask", "AttentionScoreExplainTask", "edit"],
                        help="The type of the task. On of attack, explain and edit.")

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
    parser.add_argument("--features_type", type=str, default="attention", choices=["attention"])
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

task = None
if config.task_type == "AttentionScoreAttackTask":
    task = AttentionScoreAttackTask()
    task.prepare_stage()

if config.use_wandb:
    import wandb
    wandb.init(name=exp_name, project=config.wandb_project_name, config=config)
    wandb_result_table = create_result_table(task.get_result_table_columns())

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

config.num_hidden_layers = transformer_model.config.num_hidden_layers
config.num_attention_heads = transformer_model.config.num_attention_heads
config.intermediate_size = transformer_model.config.intermediate_size

dqn = DQN(config, do_eval=False, mask_token_id=MASK_TOKEN_ID)
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

        batch.send_to_device(lm_device)

        task.before_episode(batch)
        if task.eval_no_grad:
            with torch.no_grad():
                original_outputs = transformer_model(task.organize_original_input(batch))
        else:
            original_outputs = transformer_model(task.organize_original_input(batch))
        task.process_original_output(original_outputs, batch)

        if config.do_pre_deletion:
            # Delete misclassified samples when training
            left_index = [i for i, acc in enumerate(batch.original_acc) if acc == 1]
            batch.keep_items(left_index)
            if batch.batch_size == 0:
                continue

        for game_step in range(config.max_game_steps):

            task.before_step(batch)
            if task.eval_no_grad:
                with torch.no_grad():
                    perturbed_outputs = transformer_model(task.organize_step_input(batch))
            else:
                perturbed_outputs = transformer_model(task.organize_step_input(batch))
            task.process_step_output(perturbed_outputs, batch)
            task.done_function(batch)

            dqn.store_transition(batch_size, special_tokens_mask, next_special_tokens_mask, now_features, next_features, now_game_status, next_game_status, actions, seq_length, r.rewards, r.if_success)

            success_index = [i for i in range(batch_size) if r.if_success[i].item() == 1]
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
