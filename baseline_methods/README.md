# Comparison Methods

`data_set_name` is one of `emotion`, `snli` or `sst2`.

## For Model Explanation

LIME: 

    python baseline_methods/explain_baseline_LIME.py --data_set_name emotion  --gpu_index 0  --lime_max_sampling_num 100 --use_wandb

Captum: `explain_method` is one of `FeatureAblation`, `Occlusion`, `KernelShap` or `ShapleyValueSampling`.

    python baseline_methods/explain_baseline_captum.py --data_set_name emotion --explain_method FeatureAblation --use_wandb

You can add an argument `--max_sample_num` when using `KernelShap` or `ShapleyValueSampling`. This parameter will limit the maximum number of samples (Model Query Times) for the `KernelShap` method. But for `ShapleyValueSampling` method, it will limit the maximum number of samples for each token. So the real total number of Model Query Times for one sequence should be `seq_length * max_sample_num`.


## For Adversarial Attack

The website of OpenAttack toolkit is https://github.com/thunlp/OpenAttack . We use it for evaluating all baseline methods in text adversarial attack task.

    python baseline_methods/attack_baseline_openattack.py 