# Attack baseline toolkit OpenAttack
# https://github.com/thunlp/OpenAttack

import os
import sys
sys.path.append(os.getcwd())
import logging
import argparse

import datetime
from data_utils import *
import OpenAttack as oa
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run attack baseline")

    parser.add_argument(
        "--data_set_name", type=str, default=None, help="The name of the dataset. On of emotion, snli or sst2."
    )
    parser.add_argument("--eval_test_batch_size", type=int, default=32)

config = parse_args()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger.info(f"Eval config: {config}")

dataset_config = get_dataset_config(config)
problem_type = dataset_config["problem_type"]
num_labels = dataset_config["num_labels"]
label_names = dataset_config["label_names"]
text_col_name = dataset_config["text_col_name"]
text_col_num = dataset_config["text_col_num"]
token_quantity_correction = dataset_config["token_quantity_correction"]

tokenizer = AutoTokenizer.from_pretrained(dataset_config["model_name_or_path"])

logger.info("Start loading!")
transformer_model, simulate_dataset, eval_dataset = get_dataset_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")


class NLIWrapper(oa.classifiers.Classifier):
    def __init__(self, model: oa.classifiers.Classifier):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        ref = self.context.input["hypothesis"]
        input_sents = [sent + "</s></s>" + ref for sent in input_]
        return self.model.get_prob(
            input_sents
        )

victim = oa.classifiers.TransformersClassifier(transformer_model, tokenizer, transformer_model.bert.embeddings.word_embeddings)
if config.data_set_name == "snli":
    victim = NLIWrapper(victim)


attacker_list = [
    oa.attackers.PWWSAttacker(),
    oa.attackers.GeneticAttacker(),
    oa.attackers.SCPNAttacker(),
    oa.attackers.HotFlipAttacker(),
    oa.attackers.DeepWordBugAttacker(),
    oa.attackers.TextBuggerAttacker(),
    oa.attackers.PSOAttacker(),
    oa.attackers.BERTAttacker(), 
]

# prepare for attacking
dt = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
save_file_dir = f"logs/openattack_{config.data_set_name}_{dt}"

f = open(save_file_dir, "w")
# will wait for a long time
for attacker in attacker_list:

    print(f"Start attack by {attacker.__class__.__name__}")
    attack_eval = oa.AttackEval(attacker, victim, metrics=[
        oa.metric.EditDistance(),
        oa.metric.ModificationRate()
    ])
    try:
        result = attack_eval.eval(simulate_dataset, visualize=False)
    except:
        print(f"some thing wrong! {attacker.__class__.__name__}")

    print(result)
    f.writelines(str(attacker.__class__.__name__)+"\n")
    f.writelines(str(result))
    f.writelines("\n----------\n")

f.close()
