
"""
For a $n$ length sequence, we enumerate all $2^n$ combinations. 
These samples are encoded and take the output representing vector at the [CLS] position and downscale to two dimensions using T-SNE. 
Each point in the two-dimensional space is a constructed sample and the color of the point is the classification label of this sample.
"""

# %%
import argparse
import copy
import logging
import sys
from collections import defaultdict, Counter

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import torch
from accelerate.utils import send_to_device
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data_utils import (get_dataloader_and_model, get_dataset_config,
                        get_word_masked_rate)

# %%
from dqn_model import DQN
from utils import *

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack analysis process")

    parser.add_argument("--task_type", type=str, default="attack",
                        choices=["attack", "explain"],
                        help="The type of the task. On of attack or explain.")

    parser.add_argument("--input_text", type=str)
    parser.add_argument(
        "--data_set_name", type=str, default="emotion", help="The name of the dataset. On of emotion,snli or sst2."
    )
    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["statistical_bin", "const", "random", "effective_information", "gradient", "gradient_input", "original_embedding", "input_ids"])
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    parser.add_argument("--dqn_weights_path", type=str)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--is_agent_on_GPU", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


config = parse_args()
dataset_config = get_dataset_config(config)
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
token_quantity_correction = dataset_config["token_quantity_correction"]  # the number of [CLS],[SEP] special token in an example

set_seed(config.seed)
lm_device = torch.device("cuda", config.gpu_index)

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape = input_feature_shape_dict[config.features_type]

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id
config.vocab_size = tokenizer.vocab_size

input_text = config.input_text

logger.info("Start loading!")
transformer_model, _, _ = get_dataloader_and_model(config, dataset_config, tokenizer, return_simulate_dataloader=False, return_eval_dataloader=False)
logger.info("Finish loading!")


def get_rewards(original_seq_length=None,
                original_acc=None, original_prob=None, original_logits=None, original_loss=None,
                post_acc=None, post_prob=None, post_logits=None, post_loss=None,
                game_status=None, game_step=None):

    original_seq_length = (torch.FloatTensor(original_seq_length) - token_quantity_correction)
    original_seq_length, game_status = keep_tensor_in_same_device(original_seq_length, game_status, device="cpu")

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
        if_success = torch.logical_not(post_acc.bool()).float()  # Relaxed criteria for determining the success of an attack
        # if_success = torch.logical_and(original_acc.bool(), torch.logical_not(post_acc.bool())).float() # Strictest criteria for determining the success of an attack
        # if_success = torch.logical_xor(original_acc.bool(), post_acc.bool()).float()
        delta_prob, if_success = keep_tensor_in_same_device(delta_prob, if_success, device="cpu")
        rewards = delta_prob + 10 * if_success * unmasked_token_rate

    elif config.task_type == "explain":
        if_success = (delta_prob >= config.done_threshold).float()
        delta_prob, if_success = keep_tensor_in_same_device(delta_prob, if_success, device="cpu")
        rewards = delta_prob + 10 * if_success * unmasked_token_rate

    if_done = if_success.clone()  # die or win == 1
    for i in range(unmask_token_num.size(0)):
        # mask all tokens in one example will be treated as done
        if unmask_token_num[i] == 0:
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


def one_step(transformer_model, original_pred_labels, post_batch, seq_length, config, lm_device, dqn_device):

    features_type = config.features_type
    post_batch = send_to_device(post_batch, lm_device)
    if features_type == "gradient":
        extracted_features, post_outputs = get_gradient_features(transformer_model, post_batch, original_pred_labels, times_input=False)
    elif features_type == "gradient_input":
        extracted_features, post_outputs = get_gradient_features(transformer_model, post_batch, original_pred_labels, times_input=True)
    elif features_type == "original_embedding":
        extracted_features, post_outputs = use_original_embedding_as_features(transformer_model, post_batch)
    else:
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
            elif features_type == "input_ids":
                extracted_features = post_batch["input_ids"]
            else:
                raise NotImplementedError(f"features_type {features_type} not implemented")

    post_acc, post_pred_labels, post_prob = batch_accuracy(post_outputs, original_pred_labels, device=dqn_device)
    post_logits = batch_logits(post_outputs, original_pred_labels, device=dqn.device)
    post_loss = batch_loss(post_outputs, original_pred_labels, num_labels, device=dqn_device)

    now_features = extracted_features.unsqueeze(1)

    return now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss


dqn = DQN(config, do_eval=True, mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)
exp_name = "plot_path"

transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

# Step 1: Get the original prediction
batch = tokenizer.batch_encode_plus([input_text, ],
                                    add_special_tokens=True,
                                    return_token_type_ids=True,
                                    return_special_tokens_mask=True,
                                    truncation=True, padding=True, max_length=256, return_tensors="pt")

batch_max_seq_length = len(batch["input_ids"][0])
special_tokens_mask = batch.pop("special_tokens_mask")[0]

original_input_ids = batch["input_ids"][0]
original_batch = clone_batch(batch)
original_batch = send_to_device(original_batch, lm_device)
original_outputs = transformer_model(**original_batch)

golden_labels = torch.tensor([1], dtype=torch.long)
original_acc, original_pred_labels, original_prob = batch_initial_prob(original_outputs, golden_labels, device="cpu")
print(f"original_pred_label is {label_names[original_pred_labels.item()]}, original_prob is {original_prob.item()}")

# Step 2: Obtain explain path by ddqn agent
epoch_game_steps = config.max_game_steps
game_step_progress_bar = tqdm(desc="game_step", total=epoch_game_steps)

# track the modified tokens
left_index = [i for i in range(batch_max_seq_length)]
modified_index_order = []
modified_sample_prob = [original_prob.item(),]
explain_path = [batch["input_ids"], ] # the start of the path is original input_ids

# initial batch
original_seq_length = [batch_max_seq_length, ]
seq_length = [batch_max_seq_length, ]

actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, dqn.device)
now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, batch, seq_length, config,
                                                                                       lm_device=lm_device, dqn_device=dqn.device)

is_success = False
for game_step in range(epoch_game_steps):
    game_step_progress_bar.update()

    post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, special_tokens_mask, now_features, now_game_status)
    next_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config,
                                                                                            lm_device=lm_device, dqn_device=dqn.device)
    explain_path.append(post_batch["input_ids"]) # add the new input_ids to the path
    action = actions.item()
    modified_sample_prob.append(post_prob.item())
    if config.token_replacement_strategy == "delete":
        # in delete mode, delete index from origional_index and add to modified_index_order
        origional_index = left_index[action]
        modified_index_order.append(origional_index)
        left_index.remove(origional_index)

    elif config.token_replacement_strategy == "mask":
        # in mask mode, add index to modified_index_order
        origional_index = left_index[action]
        modified_index_order.append(origional_index)
    else:
        raise ValueError("token_replacement_strategy should be delete or mask")

    r = get_rewards(original_seq_length,
                    original_acc=original_acc, original_prob=original_prob, original_logits=None, original_loss=None,
                    post_acc=post_acc, post_prob=post_prob, post_logits=None, post_loss=None,
                    game_status=next_game_status, game_step=game_step)

    if r.if_done[0].item() == 1:
        is_success = True
        print(f"Success in {game_step} steps")
        break

    now_features, now_game_status, batch = next_features, next_game_status, post_batch

    if config.token_replacement_strategy == "delete":
        seq_length = [x-1 for x in seq_length]

if not is_success:
    print(f"Not success in { epoch_game_steps } steps.")

explain_path_len = len(explain_path)
explain_path = [send_to_device(x, "cpu") for x in explain_path]
explain_path = torch.cat(explain_path, dim=0)
# convert ids to string by tokenizer
explain_path_string = tokenizer.batch_decode(explain_path, skip_special_tokens=False)
print("explain_path_string:")
for item, prob in zip(explain_path_string, modified_sample_prob):
    print(item, prob)

print("modified_index_order:", modified_index_order)
print("modified_sample_prob:", modified_sample_prob)

assert len(modified_index_order) + 1 == len(modified_sample_prob)
token_saliency = defaultdict(float)
last_prob = original_prob.item()
for token_index, prob in zip(modified_index_order, modified_sample_prob[1:]):
    token_saliency[token_index] = last_prob - prob
    last_prob = prob

print("token_saliency:")
for token_index, saliency in token_saliency.items():
    print(token_index, saliency)

token_saliency_list = [token_saliency[i] for i in range(batch_max_seq_length)]
token_saliency_list = torch.tensor(token_saliency_list, dtype=torch.float32)



# %%
# Step 3: Construct permutation samples and get all samples' output representation
permutation_mask = get_permutation_mask_matrix(batch_max_seq_length, special_tokens_mask.tolist())
permutation_mask = torch.tensor(permutation_mask, dtype=torch.bool)

all_permutation_examples = []
for i, mask in enumerate(permutation_mask):
    new_input_ids = batch["input_ids"][0].clone()
    new_input_ids[mask] = MASK_TOKEN_ID
    all_permutation_examples.append(new_input_ids)

all_permutation_examples = torch.stack(all_permutation_examples, dim=0)
print("all_permutation_examples", all_permutation_examples)
print("all_permutation_examples.shape", all_permutation_examples.shape)

batch_size = 32
all_output_reprs = []
all_label_probs = []
all_epoch = len(all_permutation_examples) // batch_size + bool(len(all_permutation_examples) % batch_size)
p = 0
for _ in tqdm(range(all_epoch)):
    if p + batch_size < len(all_permutation_examples):
        input_ids = all_permutation_examples[p:p+batch_size].clone()
        small_batch_size = batch_size
    else:
        input_ids = all_permutation_examples[p:].clone()
        small_batch_size = len(all_permutation_examples) - p
    p += small_batch_size
    small_batch = {"input_ids": input_ids,
                   "token_type_ids": batch["token_type_ids"].repeat(small_batch_size, 1),
                   "attention_mask": batch["attention_mask"].repeat(small_batch_size, 1)}
    small_batch = send_to_device(small_batch, lm_device)
    with torch.no_grad():
        outputs = transformer_model(**small_batch, output_hidden_states=True)
        pooled_hidden_states = outputs.hidden_states[-1]
        pooled_hidden_states = pooled_hidden_states[:, 0, :]
        all_output_reprs.append(pooled_hidden_states.cpu())
        probs = torch.softmax(outputs.logits, dim=-1)
        # prob of original_pred_labels
        label_prob = probs[:, original_pred_labels.item()]
        all_label_probs.append(label_prob.cpu())

all_output_reprs = torch.cat(all_output_reprs, dim=0).numpy()
all_label_probs = torch.cat(all_label_probs, dim=0).numpy()
all_permutation_examples = all_permutation_examples.numpy()

# # if need, save output_reprs and labels
# np.savez(f"logs/{exp_name}.npz",
#          all_permutation_examples=all_permutation_examples,
#          all_output_reprs=all_output_reprs, 
#          all_label_probs=all_label_probs)

# # if need, reload output_reprs and labels
# data = np.load(f"logs/{exp_name}.npz")
# all_output_reprs = data["all_output_reprs"]
# all_label_probs = data["all_label_probs"]

explain_path = explain_path.numpy()
explain_path_index = []
for i in range(explain_path_len):
    tmp = np.where((all_permutation_examples == explain_path[i]).all(axis=1))
    explain_path_index.append(tmp[0][0])
print("explain_path_index:", explain_path_index)

# %%

# sample some examples form all_permutation_examples，make sure all token in explain_path are included
sample_num = 1000
sample_index = np.random.choice(len(all_permutation_examples), sample_num, replace=False)
sample_index = np.unique(np.concatenate((sample_index, explain_path_index)))

sample_output_reprs = all_output_reprs[sample_index]
sample_label_probs = all_label_probs[sample_index]
sample_permutation_examples = all_permutation_examples[sample_index]
sample_explain_path_index = []
for i in range(explain_path_len):
    tmp = np.where((sample_permutation_examples == explain_path[i]).all(axis=1))
    sample_explain_path_index.append(tmp[0][0])

# Stpe4: Run t-SNE and plot path
print("Run t-SNE, may take a long while...")
tsne = TSNE(n_components=2, init='pca', perplexity=20, random_state=0)
X_tsne = tsne.fit_transform(sample_output_reprs)

# save t-SNE result
np.savez(f"logs/{exp_name}_tsne.npz", X_tsne=X_tsne, sample_label_probs=sample_label_probs, sample_explain_path_index=sample_explain_path_index)

# reload t-SNE result
data = np.load(f"{exp_name}_tsne.npz")
X_tsne = data["X_tsne"]
sample_label_probs = data["sample_label_probs"]
sample_explain_path_index = data["sample_explain_path_index"]

# plot the result of t-SNE in color by label
plt.figure(figsize=(10, 5))
# color adjust by label_prob
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=sample_label_probs, cmap="PuRd", s=3)
plt.colorbar(ticks= np.arange(0, 1.1, 0.1), orientation='horizontal', label='probability')

# start point of explain path
plt.scatter(X_tsne[sample_explain_path_index[0], 0], X_tsne[sample_explain_path_index[0], 1], c="r", marker="^", s=100)

# connect point with arrow in explain_path
for i in range(len(explain_path_index)-1):
    plt.arrow(X_tsne[sample_explain_path_index[i], 0], X_tsne[sample_explain_path_index[i], 1],
              X_tsne[sample_explain_path_index[i+1], 0] - X_tsne[sample_explain_path_index[i], 0],
              X_tsne[sample_explain_path_index[i+1], 1] - X_tsne[sample_explain_path_index[i], 1],
              width=0.5, head_width=2,
              length_includes_head=True, fc='r', ec='r', alpha=0.7)

plt.savefig(f"logs/{exp_name}.svg")
# plt.show()


# %%
# Step 5: saliency score map
original_prob = original_prob.cpu().item()

saliency = defaultdict(float)
last_prob = original_prob
for token_index, prob in zip(modified_index_order, modified_sample_prob):
    saliency[token_index] = last_prob - prob
    last_prob = prob
saliency_score = [saliency[i] for i in range(batch_max_seq_length)]

plt.figure(figsize=(6, 4))
# horizontal bar chart
# using diverging color map bwr, more red for positive, more blue for negative
saliency_score_array = np.array(saliency_score)
max_abs = np.max(np.abs(saliency_score_array))
saliency_score4bwr = np.array(saliency_score) * 0.5 / max_abs + 0.5
plt.barh(range(batch_max_seq_length), saliency_score, color=plt.cm.bwr(saliency_score4bwr))

for x, y in enumerate(saliency_score):
    if y == 0:
        continue
    elif y > 0:
        plt.text(y + 0.001, x, '%.3f' % y, va='center')
    else:
        plt.text(y - 0.001, x, '%.3f' % y, va='center', ha='right')
plt.yticks(range(batch_max_seq_length), tokenizer.convert_ids_to_tokens(original_input_ids))
plt.gca().invert_yaxis()

plt.xlim(-0.15, 0.75)
plt.axvline(x=0, color='k', linestyle='--')
plt.savefig(f"logs/{exp_name}_saliency.svg")
plt.savefig(f"logs/{exp_name}_saliency.png")
# plt.show()

print("done")
