

python analysis.py --data_set_name emotion --task attack --use_wandb --dqn_weights_path report_weights/emotion_attacker_1M.bin --disable_tqdm
python analysis.py --data_set_name emotion --task explain --use_wandb --dqn_weights_path report_weights/emotion_explainer_1M.bin --disable_tqdm
python analysis.py --data_set_name sst2 --task attack --use_wandb --dqn_weights_path report_weights/sst2_attacker_3M.bin --disable_tqdm
python analysis.py --data_set_name sst2 --task explainer --use_wandb --dqn_weights_path report_weights/sst2_explainer_460000.bin --disable_tqdm
python analysis.py --data_set_name snli --task attack --use_wandb --dqn_weights_path report_weights/snli_attacker_1M.bin --disable_tqdm
python analysis.py --data_set_name snli --task attack --use_wandb --dqn_weights_path report_weights/snli_explainer_300000.bin --disable_tqdm