
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

    parser.add_argument("--task_type", type=str, default="attack",
                        choices=["attack", "explain"],
                        help="The type of the task. On of attack or explain.")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2."
    )
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--use_random_matrix", action="store_true", default=False)
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--simulate_batch_size", type=int, default=32)
    parser.add_argument("--eval_test_batch_size", type=int, default=32)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project_name", type=str, default="attexplaner")
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
    table_columns = ["completed_steps", "golden_label", "original_pred_label", "post_pred_label", "delta_p", "original_input_ids", "post_batch_input_ids"]
    wandb_result_table = wandb.Table(columns=table_columns)

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id


logger.info("Start loading!")
transformer_model, simulate_dataloader, eval_dataloader = get_dataloader_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")


def get_rewards(seq_length, original_acc, original_prob, original_loss, post_acc, post_prob, post_loss, game_status, game_step):

    seq_length = torch.FloatTensor(seq_length) - token_quantity_correction
    unmusk_token_num = game_status.sum(dim=1) - token_quantity_correction
    unmusked_token_rate = unmusk_token_num / seq_length
    musked_token_num = seq_length - unmusk_token_num
    musked_token_rate = 1 - unmusked_token_rate
    delta_p = (original_prob - post_prob)

    if config.task_type == "attack":
        ifdone = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
        # post_rewards = torch.clip((post_loss - original_loss), 0) * unmusked_token_rate + 10 * ifdone * unmusked_token_rate - game_step / config.max_game_steps
        # post_rewards = (original_prob - post_prob) * unmusked_token_rate + 10 * ifdone * unmusked_token_rate - game_step / config.max_game_steps
        post_rewards = 10 * ifdone

    elif config.task_type == "explain":
        ifdone = (delta_p >= config.done_threshold).float()
        # post_rewards = torch.clip(delta_p, 0) + 10 * ifdone * unmusked_token_rate - 0.2
        post_rewards = 10 * delta_p

    return post_rewards, ifdone, delta_p, musked_token_rate, unmusked_token_rate, musked_token_num, unmusk_token_num


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
all_unmusked_token_rate = []
all_done_game_step = []  # Average Victim Model Query Times
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

transformer_model.eval()
transformer_model = send_to_device(transformer_model, lm_device)

epoch_game_steps = config.max_game_steps
progress_bar = tqdm(desc="example", total=len(eval_dataloader), disable=config.disable_tqdm)
game_step_progress_bar = tqdm(desc="game_step", total=epoch_game_steps, disable=config.disable_tqdm)

for simulate_step, simulate_batch in enumerate(eval_dataloader):

    seq_length = simulate_batch.pop("seq_length")
    if simulate_batch.get("id") is not None:
        ids = simulate_batch.pop("id")
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

    original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device=dqn.device)
    original_loss = batch_loss(original_outputs, original_pred_labels, num_labels, device=dqn.device)

    simulate_batch_size = len(seq_length)
    simulate_batch_size_at_start = len(seq_length)
    all_eval_token_length.extend(seq_length)
    all_eval_example_num += len(seq_length)
    progress_bar.update(1)
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
        rewards, ifdone, delta_p, musked_token_rate, unmusked_token_rate, \
            musked_token_num, unmusk_token_num = get_rewards(seq_length, original_acc, original_prob, original_loss,
                                                             post_acc, post_prob, post_loss, now_game_status, game_step)

        if cumulative_rewards is None:
            cumulative_rewards = rewards
        else:
            cumulative_rewards += rewards

        # Remove those completed samples
        next_simulate_batch_size, next_seq_length, next_golden_labels, next_special_tokens_mask, next_attentions, \
            next_game_status, next_simulate_batch, next_original_pred_labels, \
            next_token_word_position_map, next_cumulative_rewards, \
            next_original_acc, next_original_loss, next_original_prob, \
            left_delta_p, left_musked_token_rate, left_unmusked_token_rate, \
            removed_index = gather_unfinished_examples(ifdone, simulate_batch_size, seq_length, golden_labels, special_tokens_mask,
                                                       next_attentions, now_game_status,
                                                       simulate_batch, original_pred_labels,
                                                       token_word_position_map, cumulative_rewards,
                                                       original_acc, original_loss, original_prob,
                                                       delta_p, musked_token_rate, unmusked_token_rate)

        all_game_step_mask_rate[game_step + 1].extend(musked_token_rate.tolist())
        all_game_step_mask_token_num[game_step + 1].extend(musked_token_num.tolist())

        removed_num = len(removed_index)
        if removed_num != 0:

            all_musked_token_rate.extend(musked_token_rate[removed_index].tolist())
            all_unmusked_token_rate.extend(unmusked_token_rate[removed_index].tolist())
            all_done_game_step.extend([game_step + 1 for _ in range(removed_num)])
            attack_successful_num += removed_num
            all_cumulative_rewards.extend(cumulative_rewards[removed_index].tolist())
            all_done_step_musk_rate_score[game_step + 1].extend(musked_token_rate[removed_index].tolist())
            all_done_step_musk_token_num[game_step + 1].extend(musked_token_num[removed_index].tolist())
            all_done_step_unmusk_token_num[game_step + 1].extend(unmusk_token_num[removed_index].tolist())

            done_seq_length = [v for i, v in enumerate(seq_length) if i in removed_index]
            done_token_word_position_map = [v for i, v in enumerate(token_word_position_map) if i in removed_index]
            word_masked_rate = get_word_masked_rate(now_game_status[removed_index], done_seq_length, done_token_word_position_map)
            all_musked_word_rate.extend(word_masked_rate)

            # fidelity
            finished_index = removed_index
            fidelity_acc, _ = compute_fidelity(transformer_model, finished_index, simulate_batch,
                                               special_tokens_mask, now_game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)
            all_fidelity.extend(fidelity_acc.tolist())

            # delta_prob
            all_delta_prob.extend(delta_p[removed_index].tolist())

            if config.use_wandb:
                save_result(removed_num, completed_steps, simulate_batch["input_ids"][removed_index], post_batch["input_ids"][removed_index],
                            golden_labels[removed_index], original_pred_labels[removed_index], post_pred_labels[removed_index],
                            delta_p[removed_index], tokenizer, label_names, wandb_result_table)

        # Start the next game step
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

        if simulate_batch_size == 0:
            if game_step + 2 <= epoch_game_steps:
                for future_step in range(game_step + 2, epoch_game_steps + 1):
                    all_game_step_done_num[future_step].append(1)
            break

        completed_steps += 1

    if simulate_batch_size != 0:
        # Not successful samples
        unfinished_index = [i for i in range(simulate_batch_size)]

        all_delta_prob.extend(left_delta_p.tolist())
        all_musked_token_rate.extend(left_musked_token_rate.tolist())
        all_unmusked_token_rate.extend(left_unmusked_token_rate.tolist())

        word_masked_rate = get_word_masked_rate(next_game_status, seq_length, token_word_position_map)
        all_musked_word_rate.extend(word_masked_rate)

        fidelity_acc, _ = compute_fidelity(transformer_model, unfinished_index, simulate_batch,
                                           special_tokens_mask, next_game_status, original_pred_labels, lm_device, mask_token_id=MASK_TOKEN_ID)
        all_fidelity.extend(fidelity_acc.tolist())

        if config.use_wandb:
            save_result(removed_num, completed_steps, simulate_batch["input_ids"], post_batch["input_ids"],
                        golden_labels, original_pred_labels, post_pred_labels, delta_p, tokenizer, label_names, wandb_result_table)

    all_game_step_done_num[game_step + 1].append(1 - simulate_batch_size / simulate_batch_size_at_start)

try:
    assert attack_example_num == len(all_delta_prob) == len(all_fidelity) == \
        len(all_musked_token_rate) == len(all_unmusked_token_rate) == len(all_musked_word_rate)
except:
    print(f"""{attack_example_num} == {len(all_delta_prob)} == {len(all_fidelity)} ==
          {len(all_musked_token_rate)} == {len(all_unmusked_token_rate)}""")


logger.info("Finish eval!")

reslut = {}
reslut["Eval Example Number"] = all_eval_example_num
reslut["Average Eval Token Length"] = np.mean(all_eval_token_length)
reslut["Attack Success Rate"] = attack_successful_num / attack_example_num
reslut["Token Modification Rate"] = np.mean(all_musked_token_rate)
reslut["Word Modification Rate"] = np.mean(all_musked_word_rate)
reslut["Word Left Rate"] = np.mean(all_unmusked_token_rate)
reslut["Average Victim Model Query Times"] = np.mean(all_done_game_step)
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
