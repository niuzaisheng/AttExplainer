# AttExplainer

Code for "AttExplainer:  Explain Transformer via Attention by Reinforcement Learning".

First installation of the relevant package.

    pip install -r requirements.txt

Before you start we strongly recommend that you register a `wandb` account.
This will record graphs and curves during the experiment.
If you want, complete the login operation in your shell. Enter the following command and follow the prompts to complete the login.

    wandb login

API keys can be found in User Settings page https://wandb.ai/settings .
For more information you can refer to https://docs.wandb.ai/quickstart .

All scripts have a parameter named `data_set_name`, please select one from `emotion`, `snli` or `sst2`.

Next is how to replicat all experiments:
## For Model Explanation
### Run RL training process

If use default training config:

    python run_train_explain.py --data_set_name emotion

Other settings can be found in `run_train_explain.py`

### Run per-trained RL agent evaluation process

- emotion

    python analysis_explain.py --data_set_name emotion --dqn_weights_path report_weights/emotion_explainer_1M.bin

- sst2

    python analysis_explain.py --data_set_name sst2 --dqn_weights_path report_weights/sst2_explainer_460000.bin

- snli

    python analysis_explain.py --data_set_name snli --dqn_weights_path report_weights/snli_explainer_300000.bin
    
## For Adversarial Attack
### Run RL training process

If use default training config:

    python run_train_attack.py --data_set_name emotion

Other settings can be found in `run_train_attack.py`

### Run per-trained RL agent evaluation process

- emotion

    python analysis_attack.py --data_set_name emotion --dqn_weights_path report_weights/emotion_attacker_1M.bin

- sst2

    python analysis_attack.py --data_set_name sst2 --dqn_weights_path report_weights/sst2_attacker_460000.bin

- snli

    python analysis_attack.py --data_set_name snli --dqn_weights_path report_weights/snli_attacker_300000.bin