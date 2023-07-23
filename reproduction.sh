
# This script is used to reproduce all evaluation results of the paper.

# We will try to assign calculations to different cards and run them in the background.
# Please modify the script to fit your own environment. 
# Expecially the CUDA_VISIBLE_DEVICES and the number of processes run in same time.

# The script is divided into two parts:
# 1. Evaluate the DQN agent in different data sets and tasks.
# 2. Reproduction for Baseline Methods

# Part 1: Evaluate the DQN agents in explain scenario within different datasets.
# group1() {
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type const --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-06-10-35-46/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type random --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-06-10-40-34/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-29-03-03-07/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-09-18-41/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-09-15-02/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-09-19-08-35-25/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-29-12-55-15/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-30-04-34-54/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2023-05-05-04-14-33/dqn-200000.bin --disable_tqdm
# }

# group2() {
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-05-10-38/dqn-200000.bin --disable_tqdm &
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-05-10-21/dqn-200000.bin --disable_tqdm &
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-08-56/dqn-200000.bin --disable_tqdm &
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-08-34/dqn-200000.bin --disable_tqdm &
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-06-53/dqn-200000.bin --disable_tqdm
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-01-02-09-06-28/dqn-200000.bin --disable_tqdm &
    # python analysis_add_tracker.py --data_set_name snli --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-03-27-11-10-46/dqn-400000.bin --disable_tqdm
# }

# group3(){
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-00-23/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-00-44/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-02-39/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-05-03-03/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-09-03-33/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-01-02-09-03-52/dqn-final.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type explain --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-03-27-11-03-22/dqn-400000.bin --disable_tqdm
# }

# export CUDA_VISIBLE_DEVICES=1
# group1 > logs/emotion_analysis_add_tracker_explain.out
# export CUDA_VISIBLE_DEVICES=1
# group2 > logs/snli_analysis_add_tracker_explain.out
# export CUDA_VISIBLE_DEVICES=1
# group3 > logs/sst2_analysis_add_tracker_explain.out

# group4(){
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2023-05-05-01-33-36/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type original_embedding --use_wandb --token_replacement_strategy delete --use_ddqn --dqn_weights_path saved_weights/emotion_2023-05-05-09-16-26/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy delete --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-19-09-04-14/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-07-07-56-44/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-06-04-47-21/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-11-19-07-49-22/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2023-05-06-09-25-45/dqn-200000.bin --disable_tqdm
# }
# group4 > logs/emotion_analysis_add_tracker_delete_explain.out

# Part 2: Evaluate the DQN agents in attack scenario within different datasets.

# group5(){
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type const --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-06-11-30-25/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type random --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-06-11-31-54/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-28-06-41-36/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-30-04-32-32/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-09-04-04-27-22/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-29-03-00-01/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-17-05-29-36/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-19-07-47-51/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2023-05-05-04-19-35/dqn-200000.bin --disable_tqdm
# }
# export CUDA_VISIBLE_DEVICES=1
# group5 > logs/emotion_analysis_add_tracker_mask_attack.out

# group6(){
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-10-04-54-29/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --use_ddqn --dqn_weights_path saved_weights/emotion_2022-12-10-05-03-07/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-04-03-32-14/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-07-07-47-34/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-06-04-44-20/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-11-18-05-45-23/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2023-05-06-09-30-18/dqn-200000.bin --disable_tqdm
# }
# group6 > logs/emotion_analysis_add_tracker_delete_attack.out

# group7(){ 
#     # snli attack mask 
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-05-28/dqn-200000.bin 
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-08-18/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-11-17/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-14-24/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-15-34/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-05-10-10-16-57/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/snli_2023-03-27-11-11-24/dqn-200000.bin
# }
# group7 > logs/snli_analysis_add_tracker_mask_attack.out

# group8(){
#     # snli attack delete
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-16/dqn-200000.bin  --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-37/dqn-200000.bin  --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-18-20/dqn-200000.bin  --disable_tqdm & 
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-20-03/dqn-200000.bin  --disable_tqdm
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-22-33/dqn-200000.bin  --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-27-06/dqn-200000.bin  --disable_tqdm
#     # （在跑）python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path
# }
# export CUDA_VISIBLE_DEVICES=2
# group8 > logs/snli_analysis_add_tracker_delete_attack.out & 

# group9(){
#     # sst2 attack mask
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-01-00-53/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-01-03-02/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-01-05-33/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-01-07-57/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-10-19-25/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-10-10-01-55/dqn-200000.bin
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type mixture --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/sst2_2023-05-12-09-20-22/dqn-200000.bin 
# }
# group9 > logs/sst2_analysis_add_tracker_mask_attack.out

# group9(){ 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-23-46/dqn-200000.bin --disable_tqdm & 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-22-15/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-55-15/dqn-200000.bin --disable_tqdm & 
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-57-18/dqn-200000.bin --disable_tqdm
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-04-45/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-09-11/dqn-200000.bin --disable_tqdm
#     # （在跑）python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path
# }
# export CUDA_VISIBLE_DEVICES=3
# group9 > logs/sst2_analysis_add_tracker_delete_attack.out


# Part 3: Reward Setting Ablation Analysis
# group9(){ 
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-09-18-41/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-03-42-52/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-12-12-03-39-31/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type gradient --use_wandb --token_replacement_strategy mask --dqn_weights_path saved_weights/emotion_2022-11-17-09-36-46/dqn-200000.bin --disable_tqdm 

#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-19-08-35-25/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-12-13-12-46-40/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-12-13-12-48-40/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type explain --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-20-10-56-35/dqn-200000.bin --disable_tqdm 
    
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack  --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-04-04-27-22/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack  --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-09-14-05-58-43/dqn-200000.bin --disable_tqdm 
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack  --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-12-13-11-14-17/dqn-200000.bin --disable_tqdm &
#     python analysis_add_tracker.py --data_set_name emotion --task_type attack  --features_type statistical_bin --use_wandb --token_replacement_strategy mask --use_ddqn --dqn_weights_path saved_weights/emotion_2022-12-13-11-16-57/dqn-200000.bin --disable_tqdm 
# }
# export CUDA_VISIBLE_DEVICES=1
# group9 > logs/emotion_analysis_add_tracker_reward_setting_ablation.out

# Part4: Explain baseline methods

# func() {
#     # for j in FeatureAblation Occlusion LIME KernelShap ShapleyValueSampling IntegratedGradients DeepLift
#     for j in IntegratedGradients DeepLift
#     do
#         echo "running $1 $j"
#         if [ $j = "ShapleyValueSampling" ]
#         then
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 1 --use_wandb --disable_tqdm
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 2 --use_wandb --disable_tqdm
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --max_sample_num 5 --use_wandb --disable_tqdm
#         elif [ $j = "LIME" ] || [ $j = "KernelShap" ] || [ $j = "IntegratedGradients" ]
#         then
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 10 --disable_tqdm
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 20 --disable_tqdm
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 30 --disable_tqdm
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --max_sample_num 100 --disable_tqdm
#         else
#             python baseline_methods/explain_baseline.py --data_set_name $1 --explain_method $j --use_wandb --disable_tqdm
#         fi
#     done
# }

# export CUDA_VISIBLE_DEVICES=1
# func "emotion" > logs/emotion-3.out &
# export CUDA_VISIBLE_DEVICES=0
# func "sst2" > logs/sst2-3.out &
# export CUDA_VISIBLE_DEVICES=1
# func "snli" > logs/snli-4.out &

# Part 4: Attack baseline methods
# python baseline_methods/attack_baseline_openattack.py --data_set_name emotion --max_sample_num 100
# python baseline_methods/attack_baseline_openattack.py --data_set_name sst2 --max_sample_num 100
# python baseline_methods/attack_baseline_openattack.py --data_set_name snli --max_sample_num 100

# Part 5: Transferability study

# group10(){ 
#     ## Transferability in input_ids-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-10-04-54-29/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-10-04-54-29/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-16/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-16/dqn-200000.bin
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-23-46/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type input_ids --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-23-46/dqn-200000.bin
# }

# group11(){ 
#     ## Transferability in embedding-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-10-05-03-07/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-10-05-03-07/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-37/dqn-200000.bin 
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-10-08-26-37/dqn-200000.bin 
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-22-15/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type original_embedding --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-10-08-22-15/dqn-200000.bin
# }

# group12(){ 
#     ## Transferability in statistical_bin-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-09-04-03-32-14/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-09-04-03-32-14/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-18-20/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-18-20/dqn-200000.bin
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-55-15/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type statistical_bin --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-55-15/dqn-200000.bin
# }

# group13(){ 
#     ## Transferability in effective_information-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-07-07-47-34/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-07-07-47-34/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-20-03/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-20-03/dqn-200000.bin
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-57-18/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type effective_information --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-05-57-18/dqn-200000.bin
# }

# group14(){ 
#     ## Transferability in gradient-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-06-04-44-20/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-12-06-04-44-20/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-22-33/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-22-33/dqn-200000.bin
#     echo "-sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-04-45/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-04-45/dqn-200000.bin
# }

# group15(){ 
#     ## Transferability in gradient_input-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-11-18-05-45-23/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2022-11-18-05-45-23/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-27-06/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2022-12-08-01-27-06/dqn-200000.bin
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-09-11/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type gradient_input --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2022-12-07-06-09-11/dqn-200000.bin
# }

# group16(){ 
#     ## Transferability in mixture-attack-delete
#     echo "emotion->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2023-05-06-09-30-18/dqn-200000.bin
#     echo "emotion->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/emotion_2023-05-06-09-30-18/dqn-200000.bin
#     echo "snli->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2023-05-09-01-02-29/dqn-200000.bin
#     echo "snli->sst2"
#         python analysis_add_tracker.py --data_set_name sst2 --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/snli_2023-05-09-01-02-29/dqn-200000.bin
#     echo "sst2->emotion"
#         python analysis_add_tracker.py --data_set_name emotion --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2023-05-09-01-07-25/dqn-200000.bin
#     echo "sst2->snli"
#         python analysis_add_tracker.py --data_set_name snli --task_type attack --features_type mixture --use_wandb --token_replacement_strategy delete --dqn_weights_path saved_weights/sst2_2023-05-09-01-07-25/dqn-200000.bin
# }

# group10 > logs/reproduction_transferability_group10.log 2>&1 &
# group11 > logs/reproduction_transferability_group11.log 2>&1 &
# group12 > logs/reproduction_transferability_group12.log 2>&1 &
# group13 > logs/reproduction_transferability_group13.log 2>&1 &
# group14 > logs/reproduction_transferability_group14.log 2>&1 &
# group15 > logs/reproduction_transferability_group15.log 2>&1 &
# group16 > logs/reproduction_transferability_group16.log 2>&1 &

echo "All jobs started! Wait for all jobs to finish"
wait
echo "All jobs finished!"


tmux new-session -n snli-attack-gradient3-mask-ddqn snli_2023-05-10-10-15-34
tmux new-session -n snli-attack-gradient4-mask-ddqn snli_2023-05-10-10-16-57
