# Model interpretability training process
import sys
import os
import logging
import argparse
import numpy as np
import torch
import datetime

from tqdm.auto import tqdm
from accelerate.utils import send_to_device
from transformers import AutoTokenizer, set_seed
from data_utils import get_dataloader_and_model, get_dataset_config
from dqn_model import DQN
from utils import *
import datetime
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
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    # Game Settings
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["statistical_bin", "const", "random", "effective_information", "gradient"],)
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--do_pre_deletion", action="store_true", default=False, help="Pre-deletion of misclassified samples")
    parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])

    # Train settings
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--train_agent_on_GPU", type=bool, default=False)
    parser.add_argument("--max_train_epoch", type=int, default=1000)
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
if isinstance(text_col_name, list):
    token_quantity_correction = 3
else:
    token_quantity_correction = 2

# For DQN gather single step data into batch from replay buffer
# input_feature_shape indicates how many dimensions along sequence length
if config.features_type == "statistical_bin":
    input_feature_shape = 2  # 2D feature map
elif config.features_type == "const":
    input_feature_shape = 2  # 2D feature map
elif config.features_type == "random":
    input_feature_shape = 2  # 2D feature map
elif config.features_type == "effective_information":
    input_feature_shape = 1  # 1D feature map
elif config.features_type == "gradient":
    input_feature_shape = 2  # 2D feature map

if config.use_wandb:
    import wandb
    wandb.init(name=exp_name, project=config.wandb_project_name, config=config)
    table_columns = ["completed_steps", "golden_label", "original_pred_label", "post_pred_label", "delta_p", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id

logger.info("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")

logger.info("one example:")
for _, batch in enumerate(simulate_dataloader):
    logger.info(batch)
    break

status_dict = {} # for tqdm display some info
def update_dict(name, progress_bar, any_dict, step):
    status_dict.update(any_dict)
    progress_bar.set_postfix(status_dict)
    if config.use_wandb:
        wandb.log({name: any_dict})


def save_result(save_batch_size, completed_steps, original_input_ids, post_input_ids,
                golden_labels, original_pred_labels, post_pred_labels, delta_p, tokenizer, wandb_result_table):
    # for wandb save some info
    for i in range(save_batch_size):
        golden_label = golden_labels[i].item()
        golden_label = label_names[golden_label]
        original_pred_label = original_pred_labels[i].item()
        original_pred_label = label_names[original_pred_label]
        post_pred_label = post_pred_labels[i].item()
        post_pred_label = label_names[post_pred_label]
        item_delta_p = delta_p[i].item()
        train_batch_input_ids_example = display_ids(original_input_ids[i], tokenizer, name="train_batch_input_ids")
        post_batch_input_ids_example = display_ids(post_input_ids[i], tokenizer, name="post_batch_input_ids")
        wandb_result_table.add_data(completed_steps, golden_label, original_pred_label, post_pred_label, item_delta_p, train_batch_input_ids_example, post_batch_input_ids_example)


def get_rewards(original_seq_length, original_prob, post_acc, post_prob, game_status, game_step):

    original_seq_length = original_seq_length.to(game_status.device)
    unmusk_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmusked_token_rate = unmusk_token_num / original_seq_length
    musked_token_num = original_seq_length - unmusk_token_num
    musked_token_rate = 1 - unmusked_token_rate
    delta_p = (original_prob - post_prob)

    if config.task_type == "attack":
        ifdone = torch.logical_not(post_acc.bool()).float()
        # post_rewards = torch.clip((post_loss - original_loss), 0) * unmusked_token_rate + 10 * ifdone * unmusked_token_rate - game_step / config.max_game_steps
        post_rewards = 10 * ifdone

    elif config.task_type == "explain":
        ifdone = (delta_p >= config.done_threshold).float()
        # post_rewards = torch.clip(delta_p, 0) + 10 * ifdone * unmusked_token_rate - 0.2
        post_rewards = 10 * delta_p

    for i in range(unmusk_token_num.size(0)):
        if unmusk_token_num[i] == 1:
            ifdone[i] = 1
    
    return delta_p, post_rewards, ifdone, musked_token_rate, unmusked_token_rate


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


dqn = DQN(config, MASK_TOKEN_ID, input_feature_shape)
progress_bar = tqdm(total=config.max_train_epoch * len(simulate_dataloader), disable=config.disable_tqdm)
exp_name = "simulate"

lm_device = torch.device("cuda", config.gpu_index)
transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

if config.features_type == "gradient":
    embedding_weight_tensor = transformer_model.get_input_embeddings().weight
    embedding_weight_tensor.requires_grad_(True)

completed_steps = 0
for epoch in range(config.max_train_epoch):
    update_dict(exp_name, progress_bar, {"epoch": epoch}, completed_steps)

    for simulate_step, batch in enumerate(simulate_dataloader):

        seq_length = batch.pop("seq_length") # vailid tokens num in each sequence, a list of int 
        batch_max_seq_length = max(seq_length)
        golden_labels = batch.pop("labels") 
        special_tokens_mask = batch.pop("special_tokens_mask").bool()
        special_tokens_mask = send_to_device(special_tokens_mask, dqn.device)
        token_word_position_map = batch.pop("token_word_position_map")
        batch = send_to_device(batch, lm_device)

        with torch.no_grad():
            original_outputs = transformer_model(**batch, output_attentions=True)

        original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)
        original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

        update_dict(exp_name, progress_bar, {"original_prob": original_prob.mean().item()}, completed_steps)

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
        original_seq_length = torch.FloatTensor(seq_length) - token_quantity_correction

        actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, dqn.device)
        now_features, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, batch, seq_length, config.bins_num,
                                                                                  lm_device=lm_device, dqn_device=dqn.device, features_type=config.features_type)

        batch_done_step = []
        cumulative_rewards = 0
        now_special_tokens_mask = special_tokens_mask

        for game_step in range(config.max_game_steps):

            post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, now_special_tokens_mask, now_features, now_game_status)
            next_features, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config.bins_num,
                                                                                       lm_device=lm_device, dqn_device=dqn.device, features_type=config.features_type)
            delta_p, rewards, ifdone, musked_token_rate, unmusked_token_rate = get_rewards(original_seq_length, original_prob, post_acc, post_prob, next_game_status, game_step)

            dqn.store_transition(batch_size, now_special_tokens_mask, next_special_tokens_mask, now_features, next_features, now_game_status, next_game_status, actions, seq_length, rewards, ifdone)

            cumulative_rewards += rewards
            removed_index = [i for i in range(batch_size) if ifdone[i].item() == 1]
            if len(removed_index) != 0:
                batch_done_step.extend([game_step for _ in range(len(removed_index))])
                if completed_steps % 10 == 0:
                    finished_index = removed_index
                    fidelity_plus, _ = compute_fidelity(transformer_model, finished_index, batch,
                                                        now_special_tokens_mask, now_game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)

                    update_dict(exp_name, progress_bar, {
                        "done_rewards": rewards[removed_index].mean().item(),
                        "done_musked_token_rate": musked_token_rate[removed_index].mean().item(),
                        "done_unmusked_token_rate": unmusked_token_rate[removed_index].mean().item(),
                        "done_delta_p": delta_p[removed_index].mean().item(),
                        "done_fidelity_plus": fidelity_plus.mean().item(),
                        "done_cumulative_rewards": cumulative_rewards[removed_index].mean().item(),
                        "game_step": game_step,
                    }, step=completed_steps)

                if completed_steps % 10 == 0:
                    if config.use_wandb:
                        save_result(len(removed_index), completed_steps,
                                    batch["input_ids"][removed_index], post_batch["input_ids"][removed_index],
                                    golden_labels[removed_index], original_pred_labels[removed_index], post_pred_labels[removed_index],
                                    delta_p[removed_index], tokenizer, wandb_result_table)

            if completed_steps % 10 == 0:
                update_dict(exp_name, progress_bar, {
                    "post_loss": post_loss.mean().item(),
                    "post_acc": post_acc.mean().item(),
                    "rewards": rewards.mean().item(),
                    "unmusked_token_rate": unmusked_token_rate.mean().item(),
                    "musked_token_rate": musked_token_rate.mean().item(),
                    "ifdone": ifdone.mean().item(),
                    "delta_p": delta_p.mean().item(),
                    "game_step": game_step,
                }, step=completed_steps)


            # Remove those completed samples and parpare for next game step
            batch_size, seq_length, original_seq_length, golden_labels, now_special_tokens_mask, now_features, \
                now_game_status, batch, original_pred_labels, \
                token_word_position_map, cumulative_rewards, \
                original_acc, original_loss, original_prob = \
                gather_unfinished_examples(ifdone, batch_size, seq_length, original_seq_length, golden_labels, next_special_tokens_mask,
                                           next_features, next_game_status,
                                           post_batch, original_pred_labels,
                                           token_word_position_map, cumulative_rewards,
                                           original_acc, original_loss, original_prob)

            if config.token_replacement_strategy == "delete":
                seq_length = [ x - 1 for x in seq_length]

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
                    logger.info(f"checkpoint saved in {file_name}")

                if config.use_wandb:
                    wandb.log({"input_ids": wandb_result_table})
                    wandb_result_table = wandb.Table(columns=table_columns)

            if batch_size == 0:
                update_dict(exp_name, progress_bar, {"all_done_step": game_step}, step=completed_steps)
                break

        if batch_size != 0:
            # Can't reach finish status examples.
            if config.use_wandb:
                save_result(batch_size, completed_steps,
                            batch["input_ids"], post_batch["input_ids"],
                            golden_labels, original_pred_labels, post_pred_labels,
                            delta_p, tokenizer, wandb_result_table)

        done_rate = 1 - batch_size / batch_size_at_start
        update_dict(exp_name, progress_bar, {"average_done_step": np.mean(batch_done_step), "done_rate": done_rate}, step=completed_steps)
        progress_bar.update(1)


logger.info("Finish training!")
if config.use_wandb:
    wandb.log({"input_ids": wandb_result_table})
    wandb.finish()

with open(f"{save_file_dir}/dqn-final.bin", "wb") as f:
    torch.save(dqn.eval_net.state_dict(), f)
logger.info(f"Finish saving in {save_file_dir}")
