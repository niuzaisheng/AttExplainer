# AttExplainer

Code for "AttExplainer: Explain Transformer via Attention by Reinforcement Learning". Published in IJCAI-ECAI 2022.

First clone the code and installation of the relevant package.

    git clone https://github.com/niuzaisheng/AttExplainer.git
    cd AttExplainer
    pip install -r requirements.txt
    git clone https://github.com/niuzaisheng/per.git

Before you start we strongly recommend that you register a `wandb` account.
This will record graphs and curves during the experiment.
If you want, complete the login operation in your shell. Enter the following command and follow the prompts to complete the login.

    wandb login

API keys can be found in User Settings page https://wandb.ai/settings. For more information you can refer to https://docs.wandb.ai/quickstart .

If you want to run without logining wandb account, you can set the environment variable `WANDB_MODE=offline` to save the metrics locally, no internet required.

All scripts have a parameter named `data_set_name`, please select one from `emotion`, `snli` or `sst2`.
Please make sure the network is connected during the training or inferring process, which requires downloading datasets and models.

Next is how to replicat all experiments:
## For Model Explanation
### Run RL training process

If use default training config:

    mkdir saved_weights
    python run_train.py --data_set_name emotion --task_type explain --use_wandb

Other settings can be found in `run_train_explain.py`

### Run per-trained RL agent evaluation process

    # emotion
    python analysis.py --data_set_name emotion --task explain --use_wandb --dqn_weights_path report_weights/emotion_explainer_1M.bin

    # sst2
    python analysis.py  --data_set_name sst2 --task explain --done_threshold 0.7 --use_wandb --dqn_weights_path report_weights/sst2_explainer_460000.bin

    # snli
    python analysis.py  --data_set_name snli --task explain --use_wandb --dqn_weights_path report_weights/snli_explainer_300000.bin
    
## For Adversarial Attack
### Run RL training process

If use default training config:

    mkdir saved_weights
    nohup python run_train.py --data_set_name emotion --task_type attack --use_wandb

Other settings can be found in `run_train_attack.py`

### Run per-trained RL agent evaluation process

    # emotion
    python analysis.py --data_set_name emotion --task attack --use_wandb --dqn_weights_path report_weights/emotion_attacker_1M.bin

    # sst2
    python analysis.py --data_set_name sst2 --task attack --use_wandb --dqn_weights_path report_weights/sst2_attacker_3M.bin

    # snli
    python analysis.py --data_set_name snli --task attack --use_wandb --dqn_weights_path report_weights/snli_attacker_300000.bin

## Frequently Asked Questions

- Why are the results reported in the article different from what I ran?

    The results of the experiment depend on many factors: equipments, environments, software versions. I do not guarantee that it will be able to reproduce the exact same values in all cases. But the overall trend is constant.

