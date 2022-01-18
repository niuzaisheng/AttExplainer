
# Adversarial Attack results analysis process
import sys
import logging
import argparse
from collections import defaultdict
import torch
import numpy as np
from tqdm.auto import tqdm
from accelerate.utils import send_to_device

from transformers import AutoTokenizer, set_seed

from dqn_model import DQN_eval
from data_utils import get_dataloader_and_model, get_dataset_config, get_word_masked_rate
from utils import *

logger = logging.getLogger(__name__)


set_seed(43)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack analysis process")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--use_random_matrix", type=bool, default=False)
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--simulate_batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=32)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner")
    parser.add_argument("--disable_tqdm", type=bool, default=False)
    parser.add_argument("--discribe", type=str, default="Model attack evaluation process")

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

lm_device = torch.device("cuda", config.gpu_index)

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


def save_result(save_batch_size, completed_steps, original_input_ids, post_input_ids, gold_label, original_pred_labels, post_pred_labels, tokenizer, wandb_result_table):
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


def get_rewards(seq_length, original_acc, original_loss, original_prob, post_acc, post_loss, post_prob, game_status, game_step):

    seq_length = torch.FloatTensor(seq_length) - token_quantity_correction
    unmusk_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmusk_token_rate = unmusk_token_num / seq_length
    musked_token_num = seq_length - unmusk_token_num
    musked_token_rate = 1 - unmusk_token_rate

    ifdone = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
    post_rewards = (original_prob - post_prob) * unmusk_token_rate + 10 * ifdone * unmusk_token_rate - game_step / config.max_game_steps

    return post_rewards, ifdone, musked_token_rate, unmusk_token_rate, musked_token_num, unmusk_token_num


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


dqn = DQN_eval(config, MASK_TOKEN_ID)

exp_name = "eval"
completed_steps = 0

# Metrics
attack_successful_num = 0  # Attack Success Rate
attack_example_num = 0  # Attack Success Rate
all_musked_token_rate = []  # Word Modification Rate
all_musked_word_rate = []
all_unmusk_token_rate = []
all_done_game_step = []  # Average Victim Model Query Times
all_delta_prob = []

all_game_step_done_num = defaultdict(list)  # success rate in each game_step
all_game_step_mask_rate = defaultdict(list)  # token mask rate in each game_step
all_game_step_mask_token_num = defaultdict(list)  # mask token num in each game_step
all_done_step_musk_rate_score = defaultdict(list)  # token mask rate in each done example
all_done_step_musk_token_num = defaultdict(list)  # mask token num in each done example
all_done_step_unmusk_token_num = defaultdict(list)  # unmask token num in each done example
all_eval_token_length = []
all_eval_example_num = 0

transformer_model.eval()
transformer_model = send_to_device(transformer_model, lm_device)

epoch_game_steps = config.max_game_steps
progress_bar = tqdm(desc="example", total=len(eval_dataloader), disable=config.disable_tqdm)
game_step_progress_bar = tqdm(desc="game_step", total=epoch_game_steps, disable=config.disable_tqdm)

for simulate_step, simulate_batch in enumerate(eval_dataloader):
    seq_length = simulate_batch.pop("seq_length")
    batch_max_seq_length = max(seq_length)
    golden_labels = simulate_batch.pop("labels")
    special_tokens_mask = simulate_batch.pop("special_tokens_mask")
    special_tokens_mask = send_to_device(special_tokens_mask, dqn.device)
    token_word_position_map = simulate_batch.pop("token_word_position_map")
    simulate_batch = send_to_device(simulate_batch, lm_device)
    empty_batch = get_empty_batch(simulate_batch, special_tokens_mask, mask_token_id=MASK_TOKEN_ID)

    with torch.no_grad():
        original_outputs = transformer_model(**simulate_batch, output_attentions=True)
        empty_outputs = transformer_model(**empty_batch, output_attentions=True)

    original_acc, original_pred_labels, original_prob = batch_accuracy(original_outputs, simulate_batch["label"], device=dqn.device)
    original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

    simulate_batch_size = len(seq_length)

    simulate_batch_size_at_start = len(seq_length)
    all_eval_token_length.extend(seq_length)
    all_eval_example_num += len(seq_length)
    progress_bar.update(simulate_batch_size)
    game_step_progress_bar.reset()
    attack_example_num += simulate_batch_size
    all_game_step_mask_rate[0].append(0)
    all_game_step_mask_token_num[0].append(0)
    cumulative_rewards = None

    # initial batch
    post_batch, actions, last_game_status, action_value = dqn.initial_action(simulate_batch, special_tokens_mask, seq_length, batch_max_seq_length, lm_device)
    all_attentions, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, simulate_batch, seq_length, config.bins_num,
                                                                                lm_device=lm_device, dqn_device=dqn.device, use_random_matrix=config.use_random_matrix)
    for game_step in range(epoch_game_steps):
        game_step_progress_bar.update()
        game_step_progress_bar.set_postfix({"left_examples": simulate_batch_size})
        all_game_step_done_num[game_step].append(1 - simulate_batch_size / simulate_batch_size_at_start)

        post_batch, actions, now_game_status, action_value = dqn.choose_action_for_eval(simulate_batch, seq_length, special_tokens_mask, all_attentions, last_game_status)
        next_attentions, post_acc, post_loss, post_prob, post_pred_labels = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config.bins_num,
                                                                                     lm_device=lm_device, dqn_device=dqn.device, use_random_matrix=config.use_random_matrix)

        rewards, ifdone, musked_token_rate, unmusk_token_rate, musked_token_num, unmusk_token_num = get_rewards(seq_length,
                                                                                                                original_acc, original_loss, original_prob,
                                                                                                                post_acc, post_loss, post_prob, now_game_status, game_step)

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

        all_game_step_mask_rate[game_step + 1].extend(musked_token_rate.tolist())
        all_game_step_mask_token_num[game_step + 1].extend(musked_token_num.tolist())

        removed_num = len(removed_index)
        if removed_num != 0:
            rewards_list = rewards[removed_index]

            all_musked_token_rate.extend(musked_token_rate[removed_index].tolist())
            all_unmusk_token_rate.extend(unmusk_token_rate[removed_index].tolist())
            all_done_game_step.extend([game_step + 1 for _ in range(removed_num)])
            attack_successful_num += removed_num

            all_done_step_musk_rate_score[game_step + 1].extend(musked_token_rate[removed_index].tolist())
            all_done_step_musk_token_num[game_step + 1].extend(musked_token_num[removed_index].tolist())
            all_done_step_unmusk_token_num[game_step + 1].extend(unmusk_token_num[removed_index].tolist())

            all_cumulative_rewards.extend(cumulative_rewards[removed_index].tolist())

            done_seq_length = [v for i, v in enumerate(seq_length) if i in removed_index]
            done_token_word_position_map = [v for i, v in enumerate(token_word_position_map) if i in removed_index]
            word_masked_rate = get_word_masked_rate(now_game_status[removed_index], done_seq_length, done_token_word_position_map)
            all_musked_word_rate.extend(word_masked_rate)

            all_delta_prob.extend((original_prob[removed_index] - post_prob[removed_index]).tolist())

            if config.use_wandb:
                save_result(removed_num, completed_steps, simulate_batch["input_ids"][removed_index], post_batch["input_ids"][removed_index],
                            golden_labels[removed_index], original_pred_labels[removed_index], post_pred_labels[removed_index],
                            tokenizer, wandb_result_table)

        # Start the next game step
        simulate_batch_size = next_simulate_batch_size
        seq_length = next_seq_length
        special_tokens_mask = next_special_tokens_mask
        all_attentions = next_attentions
        game_status = next_game_status
        simulate_batch = next_simulate_batch
        original_loss, original_acc, original_pred_labels, original_prob = next_original_loss, next_original_acc, next_original_pred_labels, next_original_prob
        empty_loss = next_empty_loss
        token_word_position_map = next_token_word_position_map
        cumulative_rewards = next_cumulative_rewards

        if simulate_batch_size == 0:
            if game_step + 2 <= epoch_game_steps:
                for future_step in range(game_step + 2, epoch_game_steps + 1):
                    all_game_step_done_num[future_step].append(1)
            break

        completed_steps += 1

    if simulate_batch_size != 0:
        # No successful samples
        unfinished_index = [i for i in range(simulate_batch_size)]

        all_delta_prob.extend((original_prob - post_prob).tolist())
        all_musked_token_rate.extend(musked_token_rate.tolist())
        all_unmusk_token_rate.extend(unmusk_token_rate.tolist())

        word_masked_rate = get_word_masked_rate(game_status, seq_length, token_word_position_map)
        all_musked_word_rate.extend(word_masked_rate)

    all_game_step_done_num[game_step + 1].append(1 - simulate_batch_size / simulate_batch_size_at_start)

print("Finish eval!")

reslut = {}
reslut["Eval Example Number"] = all_eval_example_num
reslut["Average Eval Token Length"] = np.mean(all_eval_token_length)
reslut["Attack Success Rate"] = attack_successful_num / attack_example_num
reslut["Token Modification Rate"] = np.mean(all_musked_token_rate)
reslut["Word Modification Rate"] = np.mean(all_musked_word_rate)
reslut["Word Left Rate"] = np.mean(all_unmusk_token_rate)
reslut["Average Victim Model Query Times"] = np.mean(all_done_game_step)
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
