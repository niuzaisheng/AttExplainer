import json
import torch
import torch.nn.functional as F
import numpy as np

from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from functools import partial
from model import MyBertForSequenceClassification


def get_token_word_position_map(batch, tokenizer):
    input_ids = batch["input_ids"].tolist()
    res = []
    for example in input_ids:
        word_offset_map = {}
        word_index = 0
        for i, token_id in enumerate(example):
            if token_id == 0:
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            word_offset_map[i] = word_index
            if not token.startswith("##"):
                word_index += 1

        res.append(word_offset_map)
    return res


def get_word_masked_rate(batch_game_status, seq_length, batch_word_offset_maps):
    assert len(batch_game_status) == len(seq_length) == len(batch_word_offset_maps)
    batch_word_masked_rate = []
    for game_status, token_seq_length, word_offset_maps in zip(batch_game_status, seq_length, batch_word_offset_maps):
        word_num = len(set(word_offset_maps.values()))
        # word_num = word_offset_maps[token_seq_length]
        masked_word_index = set()
        for i in range(token_seq_length):
            if game_status[i] == 0:
                masked_word_index.add(word_offset_maps[i])
        batch_word_masked_rate.append(len(masked_word_index)/word_num)

    return batch_word_masked_rate


def single_sentence_data_collator(features, tokenizer, num_labels, problem_type, text_col_name="text", add_cls=True):

    first = features[0]
    batch = {}

    batch = tokenizer.batch_encode_plus([item[text_col_name] for item in features], add_special_tokens=add_cls,
                                        truncation=True, padding=True, max_length=256, return_special_tokens_mask=True, return_tensors='pt')

    token_word_position_map = get_token_word_position_map(batch, tokenizer)

    if problem_type == "single_label_classification":
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long).unsqueeze(-1)
    elif problem_type == "multi_label_classification":
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch["labels"] = F.one_hot(labels, num_classes=num_labels).float()

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["seq_length"] = torch.sum(batch["attention_mask"], dim=1).tolist()
    batch["token_word_position_map"] = token_word_position_map
    batch["special_tokens_mask"] = batch["special_tokens_mask"].bool()
    # batch["sample_index"] = [0 for item in features]
    return batch


def double_sentence_data_collator(features, tokenizer, num_labels, problem_type, text_col_name1="text", text_col_name2="text", add_cls=True):
    first = features[0]
    batch = {}

    batch = tokenizer.batch_encode_plus([(item[text_col_name1], item[text_col_name2]) for item in features],
                                        add_special_tokens=add_cls,
                                        return_token_type_ids=True,
                                        return_special_tokens_mask=True, truncation=True, padding=True, max_length=256, return_tensors='pt')
    token_word_position_map = get_token_word_position_map(batch, tokenizer)

    if problem_type == "single_label_classification":
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long).unsqueeze(-1)
    elif problem_type == "multi_label_classification":
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch["labels"] = F.one_hot(labels, num_classes=num_labels).float()

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "idx") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["seq_length"] = torch.sum(batch["attention_mask"], dim=1).tolist()
    batch["token_word_position_map"] = token_word_position_map
    batch["special_tokens_mask"] = batch["special_tokens_mask"].bool()

    return batch


def get_dataset_config(config):

    if config.data_set_name == "emotion":
        model_name_or_path = "bert-base-uncased"
        adapter_name = "AdapterHub/bert-base-uncased-pf-emotion"
        label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        problem_type = "single_label_classification"
        text_col_num = 1
        text_col_name = "text"
        config.adapter_name = adapter_name

    elif config.data_set_name == "emotion2":
        model_name_or_path = "nateraw/bert-base-uncased-emotion"
        label_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
        problem_type = "single_label_classification"
        text_col_num = 1
        text_col_name = "text"

    elif config.data_set_name == "snli":
        model_name_or_path = "textattack/bert-base-uncased-snli"
        label_names = ["entailment", "neutral", "contradiction"]
        problem_type = "single_label_classification"
        text_col_num = 2
        text_col_name = ["premise", "hypothesis"]

    elif config.data_set_name == "sst2":
        model_name_or_path = "textattack/bert-base-uncased-SST-2"
        label_names = ["negative", "positive"]
        problem_type = "single_label_classification"
        text_col_num = 1
        text_col_name = "sentence"
        token_quantity_correction = 2

    else:
        raise Exception("Wrong data_set_name")

    num_labels = len(label_names)
    if isinstance(text_col_name, list):
        token_quantity_correction = 3
    else:
        token_quantity_correction = 2

    return {
        "model_name_or_path": model_name_or_path,
        "label_names": label_names,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "text_col_num": text_col_num,
        "text_col_name": text_col_name,
        "token_quantity_correction": token_quantity_correction
    }


def get_dataloader_and_model(config, dataset_config, tokenizer, return_simulate_dataloader=True):

    simulate_dataloader = None
    model_name_or_path = dataset_config["model_name_or_path"]
    label_names = dataset_config["label_names"]
    num_labels = dataset_config["num_labels"]
    problem_type = dataset_config["problem_type"]
    text_col_num = dataset_config["text_col_num"]
    text_col_name = dataset_config["text_col_name"]

    if config.data_set_name in ["emotion"]:
        dataset = load_dataset('csv', data_files={'train': 'data/emotion/test.txt', 'eval': 'data/emotion/val.txt', 'test': 'data/emotion/test.txt'}, delimiter=";")

        def add_label(example):
            example["label"] = label_names.index(example["label_name"])
            return {"text": example["text"], "label": example["label"]}
        dataset = dataset.map(add_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]
        data_collator = partial(single_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name=text_col_name)
        if return_simulate_dataloader:
            simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.simulate_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    elif config.data_set_name in ["snli"]:
        dataset = load_dataset(config.data_set_name)
        label_dict = {0: 1, 1: 2, 2: 0}

        def fix_label(example):
            example["label"] = label_dict[example["label"]]
            return example
        dataset = dataset.filter(lambda example: example['label'] != -1).map(fix_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        data_collator = partial(double_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name1=text_col_name[0], text_col_name2=text_col_name[1])
        if return_simulate_dataloader:
            simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.simulate_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["sst2"]:
        dataset = load_dataset("glue", config.data_set_name)
        dataset = dataset.remove_columns(["idx"])
        train_dataset = dataset["train"]

        # print("## before filiter length", len(train_dataset))
        # def length_filter(exmaple):
        #     text = exmaple[text_col_name]
        #     tokens = text.split(" ")
        #     return len(tokens)>5
        # train_dataset = train_dataset.filter(length_filter)
        # print("## after filiter length", len(train_dataset))

        eval_dataset = dataset["validation"]

        data_collator = partial(single_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name=text_col_name)
        if return_simulate_dataloader:
            simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.simulate_batch_size)
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    assert teacher_model.config.num_labels == num_labels

    return teacher_model, simulate_dataloader, eval_dataloader


def get_dataset_and_model(config, dataset_config, tokenizer, return_simulate_dataloader=True):

    simulate_dataloader = None
    model_name_or_path = dataset_config["model_name_or_path"]
    label_names = dataset_config["label_names"]
    num_labels = dataset_config["num_labels"]
    problem_type = dataset_config["problem_type"]
    text_col_num = dataset_config["text_col_num"]
    text_col_name = dataset_config["text_col_name"]

    if config.data_set_name in ["emotion"]:
        dataset = load_dataset('csv', data_files={'train': 'data/emotion/test.txt', 'eval': 'data/emotion/val.txt', 'test': 'data/emotion/test.txt'}, delimiter=";")

        def add_label(example):
            example["label"] = label_names.index(example["label_name"])
            return {"text": example["text"], "label": example["label"]}
        dataset = dataset.map(add_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    elif config.data_set_name in ["snli"]:
        dataset = load_dataset(config.data_set_name)
        label_dict = {0: 1, 1: 2, 2: 0}

        def fix_label(example):
            example["label"] = label_dict[example["label"]]
            return example
        dataset = dataset.filter(lambda example: example['label'] != -1).map(fix_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["sst2"]:
        dataset = load_dataset("glue", config.data_set_name)
        dataset = dataset.remove_columns(["idx"])
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    assert teacher_model.config.num_labels == num_labels

    return teacher_model, train_dataset, eval_dataset
