#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

if [ ! -d "saved_results" ]; then
    mkdir saved_results
fi

# Here are some examples of input_text:
input_text="I am feeling more confident that we will be able to take care of this baby"
# input_text="im okay but feeling a little apprehensive as my dad has a minor operation today"
# input_text="i anticipated feeling ecstatic jubilant over the moon wired giddy"
# input_text="I'm thrilled to have been given this opportunity or I would have been devastated"

# If your are using AMD CPU, please delete `-m sklearnex` in the following function calls

func(){
    echo $@
    python -m sklearnex plot_explain_path.py --task_type $1 --input_text "$2" --data_set_name $3 \
                            --features_type $4 --dqn_weights_path $5 \
                            --bins_num 32 --token_replacement_strategy mask  \
                            --max_game_steps 100 --done_threshold 0.8 --gpu_index 0 --seed 42
}

reuse_func(){
    echo $@
    python -m sklearnex plot_explain_path.py --task_type $1 --input_text "$2" --data_set_name $3 \
                            --features_type $4 --dqn_weights_path $5 --reuse \
                            --bins_num 32 --token_replacement_strategy mask  \
                            --max_game_steps 100 --done_threshold 0.8 --gpu_index 0 --seed 42
} 

# This line will run a long time
# func explain "$input_text" emotion const saved_weights/emotion_2022-12-06-10-35-46/dqn-200000.bin
# reuse_func explain "$input_text" emotion const saved_weights/emotion_2022-12-06-10-35-46/dqn-200000.bin
# Other lines will reuse a portion of the previous results for efficiency
# reuse_func explain "$input_text" emotion random saved_weights/emotion_2022-12-06-10-40-34/dqn-200000.bin
# reuse_func explain "$input_text" emotion effective_information saved_weights/emotion_2022-11-29-03-03-07/dqn-200000.bin
reuse_func explain "$input_text" emotion gradient saved_weights/emotion_2022-12-12-09-18-41/dqn-200000.bin
# reuse_func explain "$input_text" emotion gradient_input saved_weights/emotion_2022-12-12-09-15-02/dqn-200000.bin
# reuse_func explain "$input_text" emotion statistical_bin saved_weights/emotion_2022-09-19-08-35-25/dqn-200000.bin
# reuse_func explain "$input_text" emotion input_ids saved_weights/emotion_2022-11-29-12-55-15/dqn-200000.bin
# reuse_func explain "$input_text" emotion original_embedding saved_weights/emotion_2022-11-30-04-34-54/dqn-200000.bin
# reuse_func explain "$input_text" emotion mixture saved_weights/emotion_2023-05-05-04-14-33/dqn-200000.bin
