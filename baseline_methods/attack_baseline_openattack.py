# Attack baseline toolkit OpenAttack
# https://github.com/thunlp/OpenAttack

import os
import sys

sys.path.append(os.getcwd())

import argparse
import datetime
import logging
from typing import List

import OpenAttack as oa
from OpenAttack.metric import AttackMetric
from OpenAttack.tags import TAG_English 
from transformers import AutoTokenizer

from data_utils import *

logger = logging.getLogger(__name__)

oa.DataManager.enable_cdn()

def parse_args():
    parser = argparse.ArgumentParser(description="Run attack baseline")

    parser.add_argument(
        "--data_set_name", type=str, help="The name of the dataset. On of emotion, snli or sst2."
    )
    parser.add_argument("--max_sample_num", type=int, default=100)
    
    args = parser.parse_args()
    return args

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
transformer_model, _, eval_dataset = get_dataset_and_model(config, dataset_config, tokenizer)
logger.info("Finish loading!")

class NLIWrapper(oa.classifiers.Classifier):
    def __init__(self, model: oa.classifiers.Classifier):
        self.model = model

    def get_pred(self, input_):
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        # ref = self.context.input["hypothesis"]
        # input_sents = [sent + "</s></s>" + ref for sent in input_]
        return self.model.get_prob(input_)


victim = oa.classifiers.TransformersClassifier(transformer_model, tokenizer, transformer_model.bert.embeddings.word_embeddings)
if config.data_set_name == "snli":
    victim = NLIWrapper(victim)
    def rename_column(example):
        # cat premise and hypothesis
        # ids = tokenizer.encode(example["premise"], example["hypothesis"], add_special_tokens=True)
        # tokens = tokenizer.convert_ids_to_tokens(ids)
        # example["x"] = " ".join(tokens)
        example["x"] = "[CLS] " + example["premise"] + " [SEP] " + example["hypothesis"] + " [SEP]" 
        return example
    eval_dataset = eval_dataset.map(rename_column)
elif config.data_set_name == "sst2":
    def rename_column(example):
        example["x"] = example["sentence"]
        return example
    eval_dataset = eval_dataset.map(rename_column)
else:
    def rename_column(example):
        example["x"] = example["text"]
        return example
    eval_dataset = eval_dataset.map(rename_column)

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

class TokenModification(AttackMetric):
    
    NAME = "Token Modif. Rate"
    removed_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def TAGS(self):
        return { TAG_English }
        
    def calc_score(self, tokenA : List[str], tokenB : List[str]) -> float:
        """
        Args:
            tokenA: The first list of tokens.
            tokenB: The second list of tokens.
        Returns:
            Modification rate.

        Make sure two list have the same length.
        """
        va = tokenA
        vb = tokenB
        ret = 0
        if len(va) != len(vb):
            ret = abs(len(va) - len(vb))
        mn_len = min(len(va), len(vb))
        va, vb = va[:mn_len], vb[:mn_len]
        for wordA, wordB in zip(va, vb):
            if wordA != wordB:
                ret += 1
        return ret / len(tokenA)
    
    def after_attack(self, input, adversarial_sample):
        if adversarial_sample is not None:
            x = self.tokenizer.tokenize(input["x"])
            y = self.tokenizer.tokenize(adversarial_sample)
            # remove all special tokens
            x = [t for t in x if t not in TokenModification.removed_special_tokens]
            y = [t for t in y if t not in TokenModification.removed_special_tokens]
            return self.calc_score(x, y)

        return None

class DeltaProb(AttackMetric):
    
    NAME = "Delta Prob"
    
    def __init__(self, tokenizer, victim):
        self.tokenizer = tokenizer
        self.victim = victim
    
    def after_attack(self, input, adversarial_sample):

        if adversarial_sample is not None:
            original_input = input["x"]
            adversarial_input = adversarial_sample
            original_label_id = input["label"]
            res = self.victim.get_prob([original_input, adversarial_input])
            delte_p = res[0][original_label_id] - res[1][original_label_id]
            return delte_p
        
        return None


# prepare for attacking
dt = datetime.datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
save_file_dir = f"logs/openattack_{config.data_set_name}_{dt}"
print(f"The result will be saved in {save_file_dir}")

f = open(save_file_dir, "w")
# will wait for a long time
for attacker in attacker_list:

    print(f"Start attack by {attacker.__class__.__name__}")
    attack_eval = oa.AttackEval(attacker, victim,
                                metrics=[
                                    oa.metric.EditDistance(),
                                    oa.metric.ModificationRate(), # Word Modification Rate
                                    TokenModification(tokenizer), # Token Modification Rate
                                    DeltaProb(tokenizer, victim) # Delta Prob
                                ],
                                invoke_limit=config.max_sample_num)

    result = attack_eval.eval(eval_dataset, visualize=False)

    print(result)
    f.writelines(str(attacker.__class__.__name__)+"\n")
    f.writelines(str(result))
    f.writelines("\n----------\n")

f.close()
