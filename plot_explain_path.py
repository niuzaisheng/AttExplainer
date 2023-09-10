
"""
For a $n$ length sequence, we enumerate all $2^n$ combinations. 
These samples are encoded and take the output representing vector at the [CLS] position and downscale to two dimensions using T-SNE. 
Each point in the two-dimensional space is a constructed sample and the color of the point is the classification label of this sample.
"""

# %%

import argparse
import logging
from collections import defaultdict
from hashlib import md5

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
from accelerate.utils import send_to_device
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoTokenizer, set_seed

from data_utils import get_dataloader_and_model, get_dataset_config
from dqn_model import DQN
from utils import *

logger = logging.getLogger(__name__)

# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Run model attack analysis process")

    parser.add_argument("--task_type", type=str, default="attack",
                        choices=["attack", "explain"], help="The type of the task. On of attack or explain.")

    parser.add_argument("--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion,snli or sst2.")
    parser.add_argument("--input_text", type=str, default="I am feeling more confident that we will be able to take care of this baby",)
    parser.add_argument("--reuse", action="store_true", default=False,
                        help=f"""Reuse the results from three steps: Permutation sample space, t-SNE and k-means. 
                                 Keep the representation space looks same for the same input text to compare different task settings.""",)

    parser.add_argument("--bins_num", type=int, default=32)
    parser.add_argument("--features_type", type=str, default="statistical_bin",
                        choices=["const", "random", "input_ids", "original_embedding",
                                 "statistical_bin", "effective_information",
                                 "gradient", "gradient_input", "mixture"])
    parser.add_argument("--max_game_steps", type=int, default=100)
    parser.add_argument("--done_threshold", type=float, default=0.8)
    parser.add_argument("--token_replacement_strategy", type=str, default="mask", choices=["mask", "delete"])
    parser.add_argument("--use_categorical_policy", action="store_true", default=False)

    parser.add_argument("--dqn_weights_path", type=str, default="report_weights/emotion_attacker_1M.bin")
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--is_agent_on_GPU", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

config = parse_args()
config_base64 = md5(str(config).encode("utf-8")).hexdigest()

input_text = config.input_text
input_text_base64 = md5(input_text.encode("utf-8")).hexdigest()

print(f"config: {config}")
print(f"config_base64: {config_base64}")

print(f"input_text: {input_text}")
print(f"input_text_base64: {input_text_base64}")

# %% Step 0 : Load model and dataset

dataset_config = get_dataset_config(config)
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
# the number of [CLS],[SEP] special token in an example
token_quantity_correction = dataset_config["token_quantity_correction"]

set_seed(config.seed)
lm_device = torch.device("cuda", config.gpu_index)

# Different feature extraction methods will get feature matrices of different dimensions
# For DQN gather single step data into batch from replay buffer
input_feature_shape = input_feature_shape_dict[config.features_type]

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])
MASK_TOKEN_ID = tokenizer.mask_token_id
config.vocab_size = tokenizer.vocab_size

logger.info("Start loading!")
transformer_model, _, _ = get_dataloader_and_model(config, dataset_config, tokenizer, return_simulate_dataloader=False, return_eval_dataloader=False)
logger.info("Finish loading!")

# %% Step 1 : Get the original prediction and loss

transformer_model = send_to_device(transformer_model, lm_device)
transformer_model.eval()

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
original_acc, original_pred_labels, original_prob = batch_initial_prob(
    original_outputs, golden_labels, device="cpu")
original_prob = original_prob.item()
print(f"original_pred_label is {label_names[original_pred_labels.item()]}, original_prob is {original_prob:.4f}")

# %% Step 2 : Get the explanation path

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
        # Relaxed criteria for determining the success of an attack
        if_success = torch.logical_not(post_acc.bool()).float()
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
        extracted_features, post_outputs = get_gradient_features(
            transformer_model, post_batch, original_pred_labels, times_input=False)
    elif features_type == "gradient_input":
        extracted_features, post_outputs = get_gradient_features(
            transformer_model, post_batch, original_pred_labels, times_input=True)
    elif features_type == "original_embedding":
        extracted_features, post_outputs = use_original_embedding_as_features(
            transformer_model, post_batch)
    elif features_type == "mixture":
        extracted_features, post_outputs = get_mixture_features(
            transformer_model, post_batch, original_pred_labels, seq_length, config.bins_num)
    else:
        with torch.no_grad():
            post_outputs = transformer_model(
                **post_batch, output_attentions=True)
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
                raise NotImplementedError(
                    f"features_type {features_type} not implemented")

    post_acc, post_pred_labels, post_prob = batch_accuracy(
        post_outputs, original_pred_labels, device=dqn_device)
    post_logits = batch_logits(
        post_outputs, original_pred_labels, device=dqn.device)
    post_loss = batch_loss(
        post_outputs, original_pred_labels, num_labels, device=dqn_device)

    now_features = extracted_features.unsqueeze(1)

    return now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss


dqn = DQN(config, do_eval=True, mask_token_id=MASK_TOKEN_ID, input_feature_shape=input_feature_shape)

epoch_game_steps = config.max_game_steps
game_step_progress_bar = tqdm(desc="game_step", total=epoch_game_steps)

# track the modified tokens
left_index = [i for i in range(batch_max_seq_length)]
modified_index_order = []
modified_sample_prob = [original_prob, ]
# the start of the path is original input_ids
explain_path = [original_batch["input_ids"], ]

# initial batch
original_seq_length = [batch_max_seq_length, ]
seq_length = [batch_max_seq_length, ]

actions, now_game_status = dqn.initial_action(batch, special_tokens_mask, seq_length, batch_max_seq_length, dqn.device)
now_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, batch, seq_length, config,
                                                                                       lm_device=lm_device, dqn_device=dqn.device)
is_success = False
for game_step in range(epoch_game_steps):
    game_step_progress_bar.update()
    # post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(original_batch, seq_length, special_tokens_mask, now_features, now_game_status)
    post_batch, actions, next_game_status, next_special_tokens_mask = dqn.choose_action(batch, seq_length, special_tokens_mask, now_features, now_game_status)
    next_features, post_acc, post_pred_labels, post_prob, post_logits, post_loss = one_step(transformer_model, original_pred_labels, post_batch, seq_length, config,
                                                                                            lm_device=lm_device, dqn_device=dqn.device)
    # add the new input_ids to the path
    explain_path.append(post_batch["input_ids"])
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
        seq_length = [x - 1 for x in seq_length]

if not is_success:
    print(f"Not success in { epoch_game_steps } steps.")

explain_path_len = len(explain_path)
explain_path = [send_to_device(x, "cpu") for x in explain_path]
explain_path = torch.cat(explain_path, dim=0).numpy()
# convert ids to string by tokenizer
explain_path_string = tokenizer.batch_decode(
    explain_path, skip_special_tokens=False)
print("explain_path_string:")
for item, prob in zip(explain_path_string, modified_sample_prob):
    print(item, prob)

print("modified_index_order:", modified_index_order)
print("modified_sample_prob:", modified_sample_prob)

assert len(modified_index_order) + 1 == len(modified_sample_prob)
token_saliency = defaultdict(float)
last_prob = original_prob
for token_index, prob in zip(modified_index_order, modified_sample_prob[1:]):
    token_saliency[token_index] = last_prob - prob
    last_prob = prob

print("token_saliency:")
for token_index, saliency in token_saliency.items():
    print(token_index, saliency)

# save the explain result
np.savez(f"saved_results/explain_path_{config.features_type}_{config_base64}.npz",
         config=config,
         modified_index_order=modified_index_order,
         modified_sample_prob=modified_sample_prob,
         explain_path_len=explain_path_len,
         explain_path=explain_path,
         explain_path_string=explain_path_string,
         token_saliency=token_saliency,
         original_prob=original_prob,
         allow_pickle=True)

# %% Step 3: Construct all permutation samples (2^n) and get all samples' output representation

if not config.reuse:
    permutation_mask, all_masked_rate = get_permutation_mask_matrix(batch_max_seq_length, special_tokens_mask.tolist())
    permutation_mask = torch.tensor(permutation_mask, dtype=torch.bool)

    all_permutation_examples = []
    for i, mask in enumerate(permutation_mask):
        new_input_ids = original_batch["input_ids"][0].clone()
        new_input_ids[mask] = MASK_TOKEN_ID
        all_permutation_examples.append(new_input_ids)

    all_permutation_examples = torch.stack(all_permutation_examples, dim=0)
    print("all_permutation_examples.shape", all_permutation_examples.shape)

    batch_size = 32
    all_output_reprs = []
    all_label_probs = []
    all_epoch = len(all_permutation_examples) // batch_size + bool(len(all_permutation_examples) % batch_size)
    p = 0
    for _ in tqdm(range(all_epoch)):
        if p + batch_size < len(all_permutation_examples):
            input_ids = all_permutation_examples[p:p + batch_size].clone()
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
    all_permutation_examples = all_permutation_examples.cpu().numpy()

    # if need, save all_output_reprs and labels
    np.savez(f"saved_results/reprs_{input_text_base64}.npz",
            config=config,
            all_permutation_examples=all_permutation_examples,
            all_output_reprs=all_output_reprs,
            all_label_probs=all_label_probs,
            all_masked_rate=all_masked_rate,
            allow_pickle=True)

# %% Stpe 4: Run t-SNE and k-means cluster
if not config.reuse:
    print("Running t-SNE, may take a long while...")
    tsne = TSNE(n_components=2, init='pca', perplexity=200, random_state=0)
    all_output_reprs_tsne = tsne.fit_transform(all_output_reprs)

    np.savez(f"saved_results/tsne_{input_text_base64}.npz",
            config=config,
            all_output_reprs_tsne=all_output_reprs_tsne,
            allow_pickle=True)

    print("Running k-means, may take a long while...")
    # k-means cluster in the t-SNE data and plot a arrow to the center of the cluster
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(all_output_reprs_tsne)
    cluster_centers = kmeans.cluster_centers_  # (n_clusters, 2)
    cluster_labels = kmeans.labels_  # (sample_num, )

    # save k-means result, for plotting
    np.savez(f"saved_results/kmeans_{input_text_base64}.npz",
            config=config,
            n_clusters=n_clusters,
            cluster_centers=cluster_centers,
            cluster_labels=cluster_labels,
            allow_pickle=True)

    print("Finished saving Space.")

# %% Step 5: combine all_permutation_examples and explain_path_len for plotting
# For the same input_text, to be sure that the same figure of Perturbation Examples Space, we can reuse the saved results of Step 1, 3 and 4. 
# So, here we just load the saved results, and combine them for plotting.

# 5.1 read explain_path
data = np.load(f"saved_results/explain_path_{config.features_type}_{config_base64}.npz", allow_pickle=True)
explain_path_input_text = data["config"][()].input_text
modified_index_order = data["modified_index_order"]
modified_sample_prob = data["modified_sample_prob"]
explain_path_len = data["explain_path_len"]
explain_path = data["explain_path"]
explain_path_string = data["explain_path_string"]
token_saliency = data["token_saliency"]
original_prob = data["original_prob"]

# 5.2 read all_permutation_examples
if config.reuse:
    data = np.load(f"saved_results/reprs_{input_text_base64}.npz", allow_pickle=True)
    reprs_input_text = data["config"][()].input_text
    assert reprs_input_text == explain_path_input_text
    all_permutation_examples = data["all_permutation_examples"]
    all_output_reprs = data["all_output_reprs"]
    all_label_probs = data["all_label_probs"]
    all_masked_rate = data["all_masked_rate"]

# 5.3 find the index of explain_path in all_permutation_examples
explain_path_index = []
for i in range(explain_path_len):
    tmp = np.where((all_permutation_examples == explain_path[i]).all(axis=1))
    if len(tmp[0]) == 0:
        assert f"explain_path[{i}] == {explain_path[i]} not in all_permutation_examples"
    explain_path_index.append(tmp[0][0])
print("explain_path_index:", explain_path_index)

# 5.4 read tsne and kmeans result
if config.reuse:
    data = np.load(f"saved_results/tsne_{input_text_base64}.npz", allow_pickle=True)
    all_output_reprs_tsne = data["all_output_reprs_tsne"]

    data = np.load(f"saved_results/kmeans_{input_text_base64}.npz", allow_pickle=True)
    n_clusters = data["n_clusters"]
    cluster_centers = data["cluster_centers"]
    cluster_labels = data["cluster_labels"]

    print("Finished loading Space and Path.")

explain_path_tsne = all_output_reprs_tsne[explain_path_index]

# %% Step 6: Plot path cluster token dependence perfer to the cluster center
# Voting based on cluster_labels to study the most masked words under each label
# (cluster_label, count_of_each_token_been_masked)
cluster_mask_counter = np.zeros((n_clusters, batch_max_seq_length))

for index, cluster_label in enumerate(cluster_labels):
    cluster_mask_counter[cluster_label] += (all_permutation_examples[index] == tokenizer.mask_token_id)

plt.figure(figsize=(3, 3))
ax = plt.gca()

colors = ["74BEB6", "edf6f9", "ffffff", "e29578"]
colors = [f"#{i}" for i in colors]
cmap = mcolors.LinearSegmentedColormap.from_list("n", colors)

im = ax.imshow(cluster_mask_counter[:, 1:-1], cmap=cmap, interpolation='nearest')
# for i in range(n_clusters):
#     for j in range(batch_max_seq_length):
#         plt.text(j, i, f"{cluster_mask_counter[i][j]}", ha="center", va="center", color="w")

# add text "Tokens" on the bottom of the heatmap
# plt.xlabel("Tokens")
plt.xticks(range(batch_max_seq_length - 2), tokenizer.convert_ids_to_tokens(original_input_ids[1:-1]))
plt.xticks(rotation=90)
plt.ylabel("▲ Cluster Label")
# convert cluster_name 0, 1, 2, 3 to A, B, C, D ...
cluster_name = [chr(i) for i in range(65, 65 + n_clusters)]
plt.yticks(range(n_clusters), cluster_name)
# plt.title("Frequency of Token being Masked in a Cluster")
# plt.title("Dependence of Clusters on Tokens")

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("bottom", size="5%", pad=0.05)

# add text "More dependent" on the bottom of the colorbar, add text "Less dependent" on the top of the colorbar
# cb = plt.colorbar(im, orientation='horizontal', label='Prefer', cax=cax)
# cb.set_ticks([])
# cb.ax.text(-10, 0.5, "Prefer \n to \n Keep", ha="center", va="center", color="black")
# cb.ax.text(155, 0.5, "Prefer \n to \n Delete", ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig(f"saved_results/cluster_dependence_{input_text_base64}.pdf")

# %% Step 7: Plot the Perturbtion Space

plt.figure(figsize=(3, 3))
ax = plt.gca()
colors = ["edf2fb", "e2eafc", "d7e3fc", "ccdbfd", "c1d3fe", "b6ccfe", "abc4ff"]
colors = [f"#{i}" for i in colors]
cmap = mcolors.LinearSegmentedColormap.from_list("n", colors)


# color adjust by label_prob, range from 0 to 1
im = ax.scatter(all_output_reprs_tsne[:, 0], all_output_reprs_tsne[:, 1], c=all_label_probs, cmap=cmap, s=3, vmin=0, vmax=1)

# Add three examples' annotate guidewire point to out of the figure
# example 1 is all_output_reprs_tsne[0], example 2 is all_output_reprs_tsne[-1], example 3 random select from all_output_reprs_tsne
# example 1

plt.annotate(" ", xy=(all_output_reprs_tsne[0, 0], all_output_reprs_tsne[0, 1]), xytext=(0, -23), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3'), fontsize=15)

print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[0]))

# example 2
plt.annotate(" ", xy=(all_output_reprs_tsne[-1, 0], all_output_reprs_tsne[-1, 1]), xytext=(0, -23), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3'), fontsize=15)
print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[-1]))
# example 3
random_index = 20000
print("random_index:", random_index)
print("all_permutation_examples[random_index]:", all_permutation_examples[random_index])
print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[random_index]))
plt.annotate(" ", xy=(all_output_reprs_tsne[random_index, 0], all_output_reprs_tsne[random_index, 1]), xytext=(10,-10), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3,rad=-1.35'), fontsize=15)

# no sticks in x, y axis
plt.xticks([])
plt.yticks([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)

# colorbar ticks from 0 to 1
cb = plt.colorbar(im, orientation='horizontal', cax=cax)
cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

# add a line of text upper to the colorbar
# ▲ Cluster center
# cb.ax.text(0.52, 2.5, "▲ Cluster Center, ● Start of the Path, ★ End of the Path",
#            ha="center", va="center", color="black")

# marker color set to black, add cluster label on the marker
# plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c="black", marker="^", s=50)
# for i in range(n_clusters):
#     plt.text(cluster_centers[i, 0] - 4, cluster_centers[i, 1] + 4, f"{cluster_name[i]}", ha="center", va="center", color="black", fontsize=15)


# plt.title("Perturbation Example Space")
plt.tight_layout()
plt.savefig(f"saved_results/perturb_space_{input_text_base64}.png", dpi=300)
plt.clf()

print("finished plot perturb space")

# Plot the Explain Path
# plt.figure(figsize=(3, 3))
# colors = ["edf2fb", "e2eafc", "d7e3fc", "ccdbfd", "c1d3fe", "b6ccfe", "abc4ff"]
# colors = [f"#{i}" for i in colors]
# cmap = mcolors.LinearSegmentedColormap.from_list("n", colors)

# # color adjust by label_prob, range from 0 to 1
# plt.scatter(all_output_reprs_tsne[:, 0], all_output_reprs_tsne[:, 1], c=all_label_probs, cmap=cmap, s=3, vmin=0, vmax=1, alpha=0.1)

# # no sticks in x, y axis
# plt.xticks([])
# plt.yticks([])

# # start point of explain path
# plt.scatter(explain_path_tsne[0, 0], explain_path_tsne[0, 1], c="r", marker="o", s=50)
# # end point of explain path
# plt.scatter(explain_path_tsne[-1, 0], explain_path_tsne[-1, 1], c="r", marker="*", s=50)

# # # Add step index text to each point in explain_path
# # for i in range(len(explain_path_index)):
# #     plt.text(explain_path_tsne[i, 0], explain_path_tsne[i, 1], f"{i}", ha="center", va="center", color="black", fontsize=15)

# # Connect point with arrow in explain_path
# for i in range(len(explain_path_index) - 1):
#     plt.arrow(explain_path_tsne[i, 0], explain_path_tsne[i, 1],
#                 explain_path_tsne[i + 1, 0] - explain_path_tsne[i, 0],
#                 explain_path_tsne[i + 1, 1] - explain_path_tsne[i, 1],
#                 width=0.5,
#                 length_includes_head=True, fc='r', ec='r', alpha=0.5)

# rename_features_type = {
#     'random':'random',
#     'const':'const',
#     'effective_information': "ei-attention", 
#     'gradient':"grad", 
#     'gradient_input':"grad × input",
#     'input_ids':"input-ids",
#     'original_embedding':"embedding",
#     'statistical_bin':"hist-attention",
#     'mixture':"mixture"
# }

# plt.title(rename_features_type[config.features_type])
# plt.tight_layout()
# plt.savefig(f"saved_results/perturb_space_{input_text_base64}_{config.features_type}_{config_base64}.png", dpi=300)
# plt.clf()


#%% Step 8: Plot the Perturbtion Space by masked rate

plt.figure(figsize=(3, 3))
ax = plt.gca()
colors = ['619b8a', 'a1c181', 'fcca46', 'fe7f2d', '233d4d']
colors = [f"#{i}" for i in colors]
cmap = mcolors.LinearSegmentedColormap.from_list("n", colors)

im = plt.scatter(all_output_reprs_tsne[:, 0], all_output_reprs_tsne[:, 1], c=all_masked_rate, cmap=cmap, s=3, vmin=0, vmax=1)

# Add three examples' annotate guidewire point to out of the figure
# example 1 is all_output_reprs_tsne[0], example 2 is all_output_reprs_tsne[-1], example 3 random select from all_output_reprs_tsne
# example 1
plt.annotate(" ", xy=(all_output_reprs_tsne[0, 0], all_output_reprs_tsne[0, 1]), xytext=(0, -23), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3'), fontsize=15)

print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[0]))

# example 2
plt.annotate(" ", xy=(all_output_reprs_tsne[-1, 0], all_output_reprs_tsne[-1, 1]), xytext=(0, -23), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3'), fontsize=15)
print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[-1]))
# example 3
random_index = 20000
print("tokens: ", tokenizer.convert_ids_to_tokens(all_permutation_examples[random_index]))
plt.annotate(" ", xy=(all_output_reprs_tsne[random_index, 0], all_output_reprs_tsne[random_index, 1]), xytext=(10,-10), textcoords='offset points',
             arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5, headlength=5, connectionstyle='arc3,rad=-1.35'), fontsize=15)

# no sticks in x, y axis
plt.xticks([])
plt.yticks([])

divider = make_axes_locatable(ax)
cax = divider.append_axes("bottom", size="5%", pad=0.05)

# colorbar ticks from 0 to 1
cb = plt.colorbar(im, orientation='horizontal', cax=cax)
cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

plt.tight_layout()
plt.savefig(f"saved_results/perturb_space_masked_rate_{input_text_base64}.png", dpi=300)
plt.clf()

exit()

# plt.figure(figsize=(3, 3))
# colors = ['619b8a', 'a1c181', 'fcca46', 'fe7f2d', '233d4d']
# colors = [f"#{i}" for i in colors]
# cmap = mcolors.LinearSegmentedColormap.from_list("n", colors)

# # color adjust by label_prob, range from 0 to 1
# plt.scatter(all_output_reprs_tsne[:, 0], all_output_reprs_tsne[:, 1], c=all_masked_rate, cmap=cmap, s=3, vmin=0, vmax=1, alpha=0.1)

# # # # colorbar ticks from 0 to 1
# # # cb = plt.colorbar(orientation='horizontal', label='masked rate')
# # # cb.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])

# # start point of explain path
# plt.scatter(explain_path_tsne[0, 0], explain_path_tsne[0, 1], c="r", marker="o", s=50)
# # end point of explain path
# plt.scatter(explain_path_tsne[-1, 0], explain_path_tsne[-1, 1], c="r", marker="*", s=50)

# for i in range(len(explain_path_index) - 1):
#     plt.arrow(explain_path_tsne[i, 0], explain_path_tsne[i, 1],
#                 explain_path_tsne[i + 1, 0] - explain_path_tsne[i, 0],
#                 explain_path_tsne[i + 1, 1] - explain_path_tsne[i, 1],
#                 width=0.5,
#                 length_includes_head=True, fc='r', ec='r', alpha=0.5)

# # no sticks in x, y axis
# plt.xticks([])
# plt.yticks([])

# plt.title(rename_features_type[config.features_type])
# plt.tight_layout()
# plt.savefig(f"saved_results/perturb_space_masked_rate_{input_text_base64}_{config.features_type}_{config_base64}.png", dpi=300)
# plt.clf()

# %% Step 8: Save explain_path into table

explain_path_token_list = []
for i in range(len(explain_path)):
    token_list = tokenizer.convert_ids_to_tokens(explain_path[i])
    explain_path_token_list.append(token_list[1:-1])

# save a table as latex format
df = pd.DataFrame(explain_path_token_list)
# add a column of step in first column
df.insert(0, "step", range(len(explain_path)))
# add a column of probability from modified_sample_prob in last column, 保留两位小数
df.insert(df.shape[1], "prob", [round(i, 2) for i in modified_sample_prob])

# Add a column of "nearest cluster label" in last column
sample_explain_path_nearest_cluster_label = []
for i in range(len(explain_path_index)):
    # sample_explain_path_nearest_cluster_label.append(cluster_labels[explain_path_index[i]])
    sample_explain_path_nearest_cluster_label.append(cluster_name[cluster_labels[explain_path_index[i]]])
df.insert(df.shape[1], "cluster", sample_explain_path_nearest_cluster_label)

df.replace("[MASK]", "[M]", inplace=True)

print(df)

# save the table as latex format
latex = df.to_latex(index=False, escape=False,
                    column_format="|c" * df.shape[1] + "|", na_rep="", header=False)
with open(f"saved_results/explain_path_table_{config.features_type}_{config_base64}.tex", "w") as f:
    f.write(latex)

# %%
# Step 9: saliency score map

# token_saliency

# plt.figure(figsize=(6, 4))
# # horizontal bar chart
# # using diverging color map bwr, more red for positive, more blue for negative
# saliency_score_array = np.array(saliency_score)
# max_abs = np.max(np.abs(saliency_score_array))
# saliency_score4bwr = np.array(saliency_score) * 0.5 / max_abs + 0.5
# plt.barh(range(batch_max_seq_length), saliency_score,
#          color=plt.cm.bwr(saliency_score4bwr))

# for x, y in enumerate(saliency_score):
#     if y == 0:
#         continue
#     elif y > 0:
#         plt.text(y + 0.001, x, '%.3f' % y, va='center')
#     else:
#         plt.text(y - 0.001, x, '%.3f' % y, va='center', ha='right')
# plt.yticks(range(batch_max_seq_length),
#            tokenizer.convert_ids_to_tokens(original_input_ids))
# plt.gca().invert_yaxis()

# plt.xlim(-0.15, 0.75)
# plt.axvline(x=0, color='k', linestyle='--')
# plt.savefig(f"logs/{exp_name}_saliency.svg")
# plt.savefig(f"logs/{exp_name}_saliency.png")
# # plt.show()

# print("done")
