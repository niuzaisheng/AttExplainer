# Model attack training process

import sys
import os
import logging
import argparse
import numpy as np
import torch

from tqdm.auto import tqdm
from accelerate.utils import send_to_device
from transformers import AutoTokenizer, set_seed
from data_utils import get_dataloader_and_model, get_dataset_config
from dqn_model import DQN
from utils import *
import datetime
logger = logging.getLogger(__name__)


set_seed(43)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack training process")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )

    # Dqn Settings
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--max_memory_capacity", type=int, default=100000)
    parser.add_argument("--dqn_rl", type=float, default=0.0001)
    parser.add_argument("--target_replace_iter", type=int, default=100)
    parser.add_argument("--dqn_batch_size", type=int, default=256)
    parser.add_argument("--epsilon", type=float, default=0.7)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--use_categorical_policy", type=bool, default=False)

    # Game Settings
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--use_random_matrix", type=bool, default=False)
    parser.add_argument("--done_threshold", type=float, default=0.8)

    # Train settings
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--max_train_epoch", type=int, default=1000)
    parser.add_argument("--save_step_iter", type=int, default=10000)
    parser.add_argument("--simulate_batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=32)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner")
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--discribe", type=str, default="Model model attack training process")

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

dt = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
save_file_dir = f"saved_weights/{config.data_set_name}_{dt}"

dataset_config = get_dataset_config(config)
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]
if isinstance(text_col_name, list):
    token_quantity_correction = 3
else:
    token_quantity_correction = 2

if config.use_wandb:
    import wandb
    wandb.init(project=config.wandb_project_name, config=config)
    wandb_config = wandb.config
    table_columns = ["completed_steps", "sample_label", "original_pred_label", "post_pred_label", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

print("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
print("Finish loading!")

print("one example:")
for _, batch in enumerate(simulate_dataloader):
    print(batch)
    break

status_dict = {}


def update_dict(name, progress_bar, any_dict, step):
    status_dict.update(any_dict)
    progress_bar.set_postfix(status_dict)
    if config.use_wandb:
        wandb.log({name: any_dict})


def save_result(save_batch_size, completed_steps, original_input_ids,
                post_input_ids, gold_label, original_pred_labels, post_pred_labels, tokenizer, wandb_result_table):
    for i in range(save_batch_size):
        sample_label = gold_label[i].item()
        sample_label = label_names[sample_label]
        original_pred_label = original_pred_labels[i].item()
        original_pred_label = label_names[original_pred_label]
        post_pred_label = post_pred_labels[i].item()
        post_pred_label = label_names[post_pred_label]
        train_batch_input_ids_example = display_ids(original_input_ids[i], tokenizer, name="train_batch_input_ids")
        post_batch_input_ids_example = display_ids(post_input_ids[i], tokenizer, name="post_batch_input_ids")
        wandb_result_table.add_data(completed_steps, sample_label, original_pred_label, post_pred_label, train_batch_input_ids_example, post_batch_input_ids_example)


def get_rewards(seq_length, original_loss, original_prob, post_acc, post_loss, post_prob, game_status, game_step):
    unmusk_token_num = game_status.sum(dim=1)
    unmusked_token_rate = unmusk_token_num / torch.FloatTensor(seq_length).to(unmusk_token_num.device)
    musked_token_rate = 1 - unmusked_token_rate

    ifdone = torch.logical_not(post_acc.bool()).float()
    post_rewards = torch.clip((post_loss - original_loss), 0) * unmusked_token_rate + 10 * ifdone * unmusked_token_rate - game_step / config.max_game_steps

    return post_rewards, ifdone, musked_token_rate, unmusked_token_rate
    

def one_step(transformer_model, original_pred_labels, post_batch, seq_length, bins_num, lm_device, dqn_device, use_random_matrix=False):

    post_batch = send_to_device(post_batch, lm_device)
    with torch.no_grad():
        post_outputs = transformer_model(**post_batch, output_attentions=True)
        if use_random_matrix:
            post_attention = get_random_attention_features(post_outputs, config.bins_num)
        else:
            post_attention = get_attention_features(post_outputs, post_batch["attention_mask"], seq_length, bins_num)
        post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, device=dqn_device)

        post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, device=dqn_device)

    all_attentions = post_attention.unsqueeze(1)

    return all_attentions, post_acc, post_loss, post_prob, post_pred_labels


dqn = DQN(config, mask_token_id=MASK_TOKEN_ID)
progress_bar = tqdm(total=config.max_train_epoch * len(simulate_dataloader) * config.max_game_steps, disable=config.disable_tqdm)
exp_name = "simulate"

lm_device = torch.device("cuda", config.gpu_index)
transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

completed_steps = 0
for epoch in range(config.max_train_epoch):
    update_dict(exp_name, progress_bar, {"epoch": epoch}, completed_steps)

    for simulate_step, simulate_batch in enumerate(simulate_dataloader):
        seq_length = simulate_batch.pop("seq_length")
        batch_max_seq_length = max(seq_length)
        golden_labels = simulate_batch.pop("labels")
        special_tokens_mask = simulate_batch.pop("special_tokens_mask").bool()
        special_tokens_mask = send_to_device(special_tokens_mask, dqn.device)
        token_word_position_map = simulate_batch.pop("token_word_position_map")
        simulate_batch = send_to_device(simulate_batch, lm_device)

        with torch.no_grad():
            original_outputs = transformer_model(**simulate_batch, output_attentions=True)

        original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)

        original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

        update_dict(exp_name, progress_bar, {"original_acc": original_acc.mean().item(), "original_loss": original_loss.mean().item(),}, completed_steps)

        simulate_batch_size = len(seq_length)
        simulate_batch_size, seq_length, special_tokens_mask, simulate_batch, \
            original_loss, original_acc, original_pred_labels, original_prob = \
            gather_correct_examples(original_acc, simulate_batch_size, seq_length, special_tokens_mask, simulate_batch,
                                    original_loss, original_pred_labels, original_prob)
        if simulate_batch_size == 0:
            continue

        post_batch, actions, last_game_status = dqn.initial_action(simulate_batch, special_tokens_mask, seq_length, batch_max_seq_length, dqn.device)
        all_attentions, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, simulate_batch, seq_length, config.bins_num,
                                                                                    lm_device=lm_device, dqn_device=dqn.device, use_random_matrix=config.use_random_matrix)

        batch_done_step = []
        cumulative_rewards = None

        for game_step in range(config.max_game_steps):

            post_batch, actions, now_game_status = dqn.choose_action(simulate_batch, seq_length, special_tokens_mask, all_attentions, last_game_status)
            next_attentions, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config.bins_num,
                                                                                         lm_device=lm_device, dqn_device=dqn.device, use_random_matrix=config.use_random_matrix)
            rewards, ifdone, musked_token_rate, unmusked_token_rate = get_rewards(seq_length, original_loss, original_prob,
                                                                                  post_acc, post_loss, post_prob, now_game_status, game_step)

            dqn.store_transition(simulate_batch_size, all_attentions, next_attentions, last_game_status, now_game_status, actions, seq_length, rewards, ifdone)
            
            if cumulative_rewards is None:
                cumulative_rewards = rewards
            else:
                cumulative_rewards += rewards

            # Remove those completed samples
            next_simulate_batch_size, next_seq_length, next_golden_labels, next_special_tokens_mask, next_attentions, \
                next_game_status, next_simulate_batch, next_original_pred_labels, \
                next_token_word_position_map, next_cumulative_rewards, \
                next_original_acc, next_original_loss, next_original_prob, removed_index = \
                gather_unfinished_examples(ifdone, simulate_batch_size, seq_length, golden_labels, special_tokens_mask,
                                           next_attentions, now_game_status,
                                           simulate_batch, original_pred_labels,
                                           token_word_position_map, cumulative_rewards,
                                           original_acc, original_loss, original_prob)

            if len(removed_index) != 0:
                batch_done_step.extend([game_step for _ in range(len(removed_index))])
                if completed_steps % 10 == 0:
                    update_dict(exp_name, progress_bar, {
                        "done_rewards": rewards[removed_index].mean().item(),
                        "done_musked_token_rate": musked_token_rate[removed_index].mean().item(),
                        "done_unmusked_token_rate": unmusked_token_rate[removed_index].mean().item(),
                        "done_cumulative_rewards": cumulative_rewards[removed_index].mean().item(),
                        "game_step": game_step,
                    }, step=completed_steps)

                if completed_steps % 10 == 0:
                    if config.use_wandb:
                        save_result(len(removed_index), completed_steps,
                                    simulate_batch["input_ids"][removed_index], post_batch["input_ids"][removed_index],
                                    golden_labels[removed_index], original_pred_labels[removed_index], post_pred_labels[removed_index],
                                    tokenizer, wandb_result_table)

            if completed_steps % 10 == 0:
                update_dict(exp_name, progress_bar, {
                    "post_loss": post_loss.mean().item(),
                    "post_acc": post_acc.mean().item(),
                    "rewards": rewards.mean().item(),
                    "unmusked_token_rate": unmusked_token_rate.mean().item(),
                    "musked_token_rate": musked_token_rate.mean().item(),
                    "ifdone": ifdone.mean().item(),
                    "game_step": game_step,
                }, step=completed_steps)

            # Parpare for next game step
            simulate_batch_size = next_simulate_batch_size
            seq_length = next_seq_length
            golden_labels = next_golden_labels
            special_tokens_mask = next_special_tokens_mask
            all_attentions = next_attentions
            last_game_status = next_game_status
            simulate_batch = next_simulate_batch
            original_loss, original_acc, original_pred_labels, original_prob = next_original_loss, next_original_acc, next_original_pred_labels, next_original_prob
            token_word_position_map = next_token_word_position_map
            cumulative_rewards = next_cumulative_rewards

            completed_steps += 1

            if completed_steps % 10 == 9:
                dqn_loss = dqn.learn()
                update_dict(exp_name, progress_bar, {"dqn_loss": dqn_loss}, step=completed_steps)

            if completed_steps % config.save_step_iter == 0:
                if not os.path.isdir(save_file_dir):
                    os.mkdir(save_file_dir)
                file_name = f"{save_file_dir}/dqn-{completed_steps}.bin"
                with open(file_name, "wb") as f:
                    torch.save(dqn.eval_net.state_dict(), f)
                    print(f"checkpoint saved in {file_name}")

                if config.use_wandb:
                    wandb.log({"input_ids": wandb_result_table})
                    wandb_result_table = wandb.Table(columns=table_columns)

            if simulate_batch_size == 0:
                update_dict(exp_name, progress_bar, {"all_done_step": game_step}, step=completed_steps)
                break

        update_dict(exp_name, progress_bar, {"average_done_step": np.mean(batch_done_step), }, step=completed_steps)


print("Finish training!")
if config.use_wandb:
    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()

with open(f"{save_file_dir}/dqn-final.bin", "wb") as f:
    torch.save(dqn.eval_net.state_dict(), f)
print(f"Finish saving in {save_file_dir}")
