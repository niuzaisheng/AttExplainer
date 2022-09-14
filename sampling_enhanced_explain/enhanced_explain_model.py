
from captum.attr._core.kernel_shap import *
from captum.attr._utils.common import _format_input_baseline

from dqn_model import DQN
from utils import *


class EnhancedKernelShap(KernelShap):

    def __init__(self, forward_func: Callable, transformer_model, dqn_model: DQN,
                 features_type: str = "statistical_bin", bins_num: int = 32) -> None:
        super().__init__(forward_func)
        self.transformer_model = transformer_model
        self.dqn_model = dqn_model
        self.features_type = features_type
        self.bins_num = bins_num
        if features_type == "gradient":
            self.embedding_weight_tensor = transformer_model.get_input_embeddings().weight
            self.embedding_weight_tensor.requires_grad_(True)

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        token_type_ids: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        attention_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        special_tokens_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:

        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = (num_interp_features - 1) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            special_tokens_mask=special_tokens_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def kernel_shap_perturb_generator(
        self, original_inp: Union[Tensor, Tuple[Tensor, ...]], **kwargs
    ) -> Generator[Tensor, None, None]:
        r"""
        Perturbations are sampled by the following process:
         - Choose k (number of selected features), based on the distribution
                p(k) = (M - 1) / (k * (M - k))
            where M is the total number of features in the interpretable space
         - Randomly select a binary vector with k ones, each sample is equally
            likely. This is done by generating a random vector of normal
            values and thresholding based on the top k elements.

         Since there are M choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight,
         defined as:
         k(M, k) = (M - 1) / (k * (M - k) * (M choose k))
        """
        assert (
            "num_select_distribution" in kwargs and "num_interp_features" in kwargs
        ), (
            "num_select_distribution and num_interp_features are necessary"
            " to use kernel_shap_perturb_func"
        )
        if isinstance(original_inp, Tensor):
            device = original_inp.device
        else:
            device = original_inp[0].device
        num_features = kwargs["num_interp_features"]
        token_type_ids = kwargs["token_type_ids"]
        attention_mask = kwargs["attention_mask"]
        special_tokens_mask = kwargs["special_tokens_mask"]

        batch = {
            "input_ids": original_inp,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "special_tokens_mask": special_tokens_mask
        }

        yield torch.zeros(1, num_features, device=device, dtype=torch.long)
        game_status = torch.ones(1, num_features, device=device, dtype=torch.long)
        yield game_status
        if isinstance(num_features,int):
            seq_length = [ num_features ]
        elif isinstance(num_features,list):
            seq_length = num_features

        while True:
            now_features = self.get_features(batch, seq_length)
            batch, actions, game_status, special_tokens_mask = self.dqn_model.choose_action(batch, seq_length, special_tokens_mask, now_features, game_status)
            yield game_status

    def get_features(self, post_batch, seq_length):

        post_batch = send_to_device(post_batch, self.transformer_model.device)
        if self.features_type != "gradient":
            with torch.no_grad():
                post_outputs = self.transformer_model(**post_batch, output_attentions=True)
                if self.features_type == "statistical_bin":
                    extracted_features = get_attention_features(post_outputs, post_batch["attention_mask"], seq_length, self.bins_num)
                elif self.features_type == "const":
                    extracted_features = get_const_attention_features(post_outputs, self.bins_num)
                elif self.features_type == "random":
                    extracted_features = get_random_attention_features(post_outputs, self.bins_num)
                elif self.features_type == "effective_information":
                    extracted_features = get_EI_attention_features(post_outputs, seq_length)

        else:
            post_outputs = self.transformer_model(**post_batch, output_attentions=True)
            extracted_features = get_gradient_features(post_outputs, seq_length, post_batch["input_ids"], self.embedding_weight_tensor)
            self.embedding_weight_tensor.grad.zero_()

        now_features = extracted_features.unsqueeze(1)

        return now_features
