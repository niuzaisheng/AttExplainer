
# This script is used to reproduce all evaluation results of the paper.

# We will try to assign calculations to different cards and run them in the background.
# Please feel free to modify the script to fit your own environment. 
# Expecially the CUDA_VISIBLE_DEVICES and the number of processes run in same time.

# The script is divided into two parts:
# 1. Evaluate the DQN agent in different data sets and tasks.
# 2. Reproduction for Baseline Methods

# Part 1: Evaluate the DQN agent in different data sets and tasks.
# group1() {
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-29-03-03-07/dqn-200000.bin --disable_tqdm & 
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-09-18-41/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-09-15-02/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-09-19-08-35-25/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-29-12-55-15/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-30-04-34-54/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2023-03-22-02-59-57/dqn-400000.bin --disable_tqdm
# }

# group2() {
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-05-10-38/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-05-10-21/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-08-56/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-08-34/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-06-53/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-06-28/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-03-27-11-10-46/dqn-400000.bin --disable_tqdm
# }

# group3(){
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-09-03-52/dqn-final.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-09-03-33/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-03-03/dqn-200000.bin --disable_tqdm&
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-02-39/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-00-44/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-00-23/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-03-27-11-03-22/dqn-400000.bin --disable_tqdm
# }

# export CUDA_VISIBLE_DEVICES=0
# group1 > logs/emotion_analysis_add_tracker.out &
# export CUDA_VISIBLE_DEVICES=3
# group2 > logs/snli_analysis_add_tracker.out &
# export CUDA_VISIBLE_DEVICES=1
# group3 > logs/sst2_analysis_add_tracker.out &


# Part 2: Explain baseline methods

func() {
    for j in FeatureAblation Occlusion LIME KernelShap ShapleyValueSampling IntegratedGradients DeepLift
    do
        echo "running $1 $j"
        if [ $j = "ShapleyValueSampling" ]
        then
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 1 --use_wandb --disable_tqdm
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 2 --use_wandb --disable_tqdm
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 5 --use_wandb --disable_tqdm
        elif [ $j = "LIME" ] || [ $j = "KernelShap" ] || [ $j = "IntegratedGradients" ]
        then
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 10 --disable_tqdm
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 20 --disable_tqdm
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 30 --disable_tqdm
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 100 --disable_tqdm
        else
            python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --disable_tqdm
        fi
    done
}

export CUDA_VISIBLE_DEVICES=0
func "emotion" > logs/emotion-3.out &
export CUDA_VISIBLE_DEVICES=2
func "sst2" > logs/sst2-3.out &
export CUDA_VISIBLE_DEVICES=2
func "snli" > logs/snli-4.out &

echo "All jobs started! Wait for all jobs to finish"
wait
echo "All jobs finished!"
