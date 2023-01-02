# Comparison Methods

`data_set_name` is one of `emotion`, `snli` or `sst2`.

## For Model Explanation

`explain_method` is one of `FeatureAblation`, `Occlusion`, `KernelShap` , `ShapleyValueSampling`, `LIME`, `IntegratedGradients` or `DeepLift`.

    python baseline_methods/explain_baseline.py --data_set_name emotion --explain_method FeatureAblation --use_wandb

You can add an argument `--max_sample_num` when using `KernelShap`,`ShapleyValueSampling`, `LIME` and `IntegratedGradients`. This parameter will limit the maximum number of samples (Model Query Times) for explanation methods. But there is an exception for `ShapleyValueSampling` method, we cannot limit the total query times, `--max_sample_num` will limit the maximum sampling number for *each single token*. So the real total number of total Model Query Times for one sequence is `seq_length * max_sample_num`.

    python baseline_methods/explain_baseline.py --data_set_name snli --explain_method ShapleyValueSampling --max_sample_num 5 --use_wandb

## For Adversarial Attack

The website of OpenAttack toolkit is https://github.com/thunlp/OpenAttack . We use it for evaluating all baseline methods in text adversarial attack task.

    python baseline_methods/attack_baseline_openattack.py --data_set_name emotion --max_sample_num 100

    