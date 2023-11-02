import abc
import math

import torch
import torch.nn as nn

from data_structure import F, BatchTensorSet, DataStructure, DataType


# replace_forward BertSelfAttention.forward to add attention_perturbation when inference
def new_forward_func(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    print("Warining! Replace success! Inference with attention_perturbation!")
    mixed_query_layer = self.query(hidden_states)

    # If this is instantiated as a cross-attention module, the keys
    # and values come from an encoder; the attention mask needs to be
    # such that the encoder's padding tokens are not attended to.
    is_cross_attention = encoder_hidden_states is not None

    if is_cross_attention and past_key_value is not None:
        # reuse k,v, cross_attentions
        key_layer = past_key_value[0]
        value_layer = past_key_value[1]
        attention_mask = encoder_attention_mask
    elif is_cross_attention:
        key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
        value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
        attention_mask = encoder_attention_mask
    elif past_key_value is not None:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
        value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
    else:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

    query_layer = self.transpose_for_scores(mixed_query_layer)

    if self.is_decoder:
        # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        # Further calls to cross_attention layer can then reuse all cross-attention
        # key/value_states (first "if" case)
        # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        # all previous decoder key/value_states. Further calls to uni-directional self-attention
        # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        # if encoder bi-directional self-attention `past_key_value` is always `None`
        past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    if getattr(self, "attention_perturbation", None) is not None:
        attention_scores = attention_scores + self.attention_perturbation

    # Normalize the attention scores to probabilities.
    attention_probs = nn.Softmax(dim=-1)(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


class ABCTask(abc.ABC):

    tensor_set_blueprint = [
        "id", "text", "token_word_position_map",
        "original_seq_length", "golden_label", 
        "input_ids", "attention_mask", "special_tokens_mask", 
        "original_logits", "original_pred_label", "original_prob", "original_acc", "original_loss",
        "perturbed_logits", "perturbed_pred_label", "perturbed_prob", "perturbed_acc", "perturbed_loss",
        "delta_logits", "delta_prob", "delta_acc", "delta_loss",
        "step_reward", "game_step", "episode_reward",
        "if_done", "if_success", "if_timeout",
    ]

    transformer_model_input_dict = ["input_ids", "attention_mask"]
    eval_no_grad = True

    def __init__(self):
        self.num_labels = 2

    def prepare_stage(self):
        pass

    def get_result_table_columns(self):
        result_table_columns = []
        for name in self.tensor_set_blueprint:
            if F(name).need_log:
                result_table_columns.append(name)
        return result_table_columns

    def before_episode(self, batch:BatchTensorSet):
        batch.game_step = torch.zeros(batch.batch_size, dtype=torch.long)

        batch.if_done = torch.zeros(batch.batch_size, dtype=torch.bool)
        batch.if_success = torch.zeros(batch.batch_size, dtype=torch.bool)
        batch.if_timeout = torch.zeros(batch.batch_size, dtype=torch.bool)

        batch.step_reward = torch.zeros(batch.batch_size, dtype=torch.float)
        batch.episode_reward = torch.zeros(batch.batch_size, dtype=torch.float)


    def organize_original_input(self, batch:BatchTensorSet) -> dict: 
        input_dict = {}
        for name in self.transformer_model_input_dict:
            input_dict[name] = batch[name]
        return input_dict
    
    def process_original_output(self, model_output:dict, batch:BatchTensorSet):

        original_logits = model_output.logits.detach().cpu()
        original_pred_label = original_logits.argmax(1).unsqueeze(-1)
        original_label_ref = original_pred_label
        original_prob = torch.softmax(original_logits, dim=-1)
        original_prob = torch.gather(original_prob, 1, original_label_ref).reshape(-1)
        original_acc = (original_label_ref == batch.golden_label).float().reshape(-1)
        original_loss = nn.CrossEntropyLoss(reduction="none")(original_logits.view(-1, self.num_labels), original_label_ref.view(-1))

        batch.original_logits = original_logits
        batch.original_pred_label = original_pred_label
        batch.original_prob = original_prob
        batch.original_acc = original_acc
        batch.original_loss = original_loss

    def before_step(self, batch:BatchTensorSet):
        pass

    def organize_step_input(self, batch:BatchTensorSet) -> dict:
        input_dict = {}
        for name in self.transformer_model_input_dict:
            input_dict[name] = batch[name]
        return input_dict
    
    def process_step_output(self, model_output:dict, batch:BatchTensorSet):

        perturbed_logits = model_output.logits.detach().cpu()
        perturbed_pred_label = perturbed_logits.argmax(1).unsqueeze(-1)

        perturbed_label_ref = batch.original_pred_label

        perturbed_prob = torch.softmax(perturbed_logits, dim=-1)
        perturbed_prob = torch.gather(perturbed_prob, 1, perturbed_label_ref).reshape(-1)
        perturbed_acc = (perturbed_label_ref == perturbed_pred_label).float().reshape(-1)
        perturbed_loss = nn.CrossEntropyLoss(reduction="none")(perturbed_logits.view(-1, self.num_labels), perturbed_label_ref.view(-1))

        batch.perturbed_logits = perturbed_logits
        batch.perturbed_pred_label = perturbed_pred_label
        batch.perturbed_prob = perturbed_prob
        batch.perturbed_acc = perturbed_acc
        batch.perturbed_loss = perturbed_loss

        batch.delta_logits = batch.original_logits - perturbed_logits
        batch.delta_prob = batch.original_prob - perturbed_prob
        batch.delta_acc = torch.logical_xor(batch.original_acc.bool(), perturbed_acc.bool()).float()
        batch.delta_loss = batch.original_loss - perturbed_loss

    def after_step(self):
        pass

    def after_success(self):
        pass

    def after_fail(self):
        pass

    def after_done(self):
        pass

    def on_log(self):
        pass


class AttentionScoreAttackTask(ABCTask):
    tensor_set_blueprint = [
        "id", "text", "token_word_position_map",
        "original_seq_length", "golden_label", 
        "input_ids", "attention_mask", "special_tokens_mask", 
        "original_logits", "original_pred_label", "original_prob", "original_acc", "original_loss",
        "perturbed_logits", "perturbed_pred_label", "perturbed_prob", "perturbed_acc", "perturbed_loss",
        "delta_logits", "delta_prob", "delta_acc", "delta_loss",
        "original_attention", "perturbed_attention",
        "step_reward", "game_step", "episode_reward",
        "if_done", "if_success", "if_timeout",
    ]

    tensor_set_class_name = "AttentionScoreAttackTensorSet"
    eval_no_grad = True

    def __init__(self, config):
        # register_data_structure
        F.register(DataStructure(DataType.Tensor4D, "original_attention", dimension_names=["layer_num", "head_num", "seq_len1", "seq_len2", ], padding_dimensions=[2, 3]))
        F.register(DataStructure(DataType.Tensor4D, "perturbed_attention", dimension_names=["layer_num", "head_num", "seq_len1", "seq_len2", ], padding_dimensions=[2, 3]))
        self.item_tensor_set_class, self.batch_tensor_set_class = F.new_tensor_set_class(self.tensor_set_class_name, self.tensor_set_blueprint)
        
        # prepare some internal variables
        self._module_dict = {} # module instance -> layer index
        self.config = config

    def prepare_stage(self):
        from transformers.models.bert.modeling_bert import BertSelfAttention
        BertSelfAttention.forward = new_forward_func

    def organize_original_input(self, batch:BatchTensorSet) -> dict: 
        input_dict = {"output_attentions": True}
        for name in self.transformer_model_input_dict:
            input_dict[name] = batch[name]
        return input_dict

    def process_original_output(self, model_outputs:dict, batch:BatchTensorSet):
        batch_size = batch.batch_size
        batch_attentions = model_outputs.attentions # a layer_num list of  [batch_size, head_num, seq_length, seq_length]
        # cat into [batch_size, layer_num, head_num, seq_length, seq_length]
        batch_attention = torch.stack(batch_attentions, dim=1) # [batch_size, layer_num, head_num, seq_length, seq_length]
        batch.original_attention = batch_attention.detach().cpu()

        self.action_space = batch.original_attention.shape[1:]
        batch.action = torch.zeros((batch_size, *self.action_space)) # 0 is visible to the model, 1 is invisible to the model.
        batch.game_status = torch.zeros((batch_size, *self.action_space)) # 0 is not masked, 1 is masked.

        super().process_original_output(batch, batch)

    def before_step(self, batch:BatchTensorSet):
        if batch.action is not None:
            # attention_perturbation is combined with action and game_status
            # update game_status: game_status is flipped if action is 1
            batch.game_status = (batch.game_status.bool() ^ batch.action.bool()).float()
            # attention_perturbation is -inf if game_status is 1
            # attention_perturbation is 0 if game_status is 0
            attention_perturbation = batch.game_status * float("-inf")

            for module, layer_index in self._module_dict.items():
                module.attention_perturbation = attention_perturbation[:, layer_index, :, :].unsqueeze(1)

    def organize_step_input(self, batch:BatchTensorSet) -> dict:
        input_dict = {"output_attentions": True}
        for name in self.transformer_model_input_dict:
            input_dict[name] = batch[name]
        return input_dict
    
    def reward_function(self, batch:BatchTensorSet):
        batch.step_reward = batch.perturbed_prob - batch.original_prob
        batch.episode_reward += batch.step_reward

    def process_step_output(self, perturbed_outputs:dict, batch:BatchTensorSet):
        batch_attentions = perturbed_outputs.attentions # a layer_num list of  [batch_size, head_num, seq_length, seq_length]
        batch_attention = torch.stack(batch_attentions, dim=1).detach().cpu()
        batch.perturbed_attention = batch_attention
        batch.now_features = batch_attention # now_features is used for DQN network input
        super().process_step_output(perturbed_outputs, batch)
        self.reward_function(batch)

    def done_function(self, batch:BatchTensorSet):
        batch.if_success = (batch.perturbed_pred_label == batch.golden_label).bool()
        batch.if_timeout = (batch.game_step >= self.config.max_game_steps).bool()
        # if_done = if_success | if_timeout
        batch.if_done = batch.if_success | batch.if_timeout

    def detach_hook_from_model(self):
        for module, layer_index in self._module_dict.items():
            module.attention_perturbation = None
