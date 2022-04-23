## Comparison Methods

`data_set_name` is one of `emotion`, `snli` or `sst2`.


# For Model Explanation
- LIME: 

    python baseline_methods/explain_baseline_LIME.py --data_set_name emotion  --gpu_index 0  --lime_max_sampling_num 100 --use_wandb

- anchor: 

    python baseline_methods/explain_baseline_anchor.py --data_set_name emotion  --gpu_index 0 --use_wandb

- captum: `explain_method` is one of `FeatureAblation`, `Occlusion`, `KernelShap` or `ShapleyValueSampling`.

    python baseline_methods/explain_baseline_captum.py --data_set_name emotion --explain_method FeatureAblation --gpu_index 0 --use_wandb

# For Adversarial Attack

    python baseline_methods/attack_baseline_openattack.py 