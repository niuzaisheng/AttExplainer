
from functools import partial

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader

import transformers
from language_model import MyBertForSequenceClassification


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


def single_sentence_data_collator(features, tokenizer, num_labels, problem_type, text_col_name="text"):

    first = features[0]
    batch = {}

    batch = tokenizer.batch_encode_plus([item[text_col_name] for item in features],
                                        add_special_tokens=True,
                                        return_token_type_ids=True,
                                        return_special_tokens_mask=True,
                                        truncation=True, padding=True, max_length=256, return_tensors='pt')

    token_word_position_map = get_token_word_position_map(batch, tokenizer)

    if problem_type == "single_label_classification":
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long).unsqueeze(-1)

    elif problem_type == "multi_label_classification":
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch["labels"] = F.one_hot(labels, num_classes=num_labels).float()

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["seq_length"] = torch.sum(batch["attention_mask"], dim=1).tolist()
    batch["token_word_position_map"] = token_word_position_map
    batch["special_tokens_mask"] = batch["special_tokens_mask"].bool()
    if "id" in first.keys():
        batch["id"] = [item["id"] for item in features]

    return batch


def double_sentence_data_collator(features, tokenizer, num_labels, problem_type, text_col_name1="text1", text_col_name2="text2"):
    first = features[0]
    batch = {}

    batch = tokenizer.batch_encode_plus([(item[text_col_name1], item[text_col_name2]) for item in features],
                                        add_special_tokens=True,
                                        return_token_type_ids=True,
                                        return_special_tokens_mask=True,
                                        truncation=True, padding=True, max_length=256, return_tensors='pt')
    token_word_position_map = get_token_word_position_map(batch, tokenizer)

    if problem_type == "single_label_classification":
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long).unsqueeze(-1)
    elif problem_type == "multi_label_classification":
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch["labels"] = F.one_hot(labels, num_classes=num_labels).float()

    for k, v in first.items():
        if k not in ("label", "label_ids", "idx") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    batch["seq_length"] = torch.sum(batch["attention_mask"], dim=1).tolist()
    batch["token_word_position_map"] = token_word_position_map
    batch["special_tokens_mask"] = batch["special_tokens_mask"].bool()
    if "id" in first.keys():
        batch["id"] = [item["id"] for item in features]

    return batch


# For processing eraser_esnli dataset

def get_tokenized_sentence(word_list, tokenizer):
    word2token_map = {}  # word id -> token span
    tokens = []
    token_id = 0
    for word_index, word in enumerate(word_list):
        token = tokenizer.encode(word, add_special_tokens=False)
        tokens.extend(token)
        word2token_map[word_index] = (token_id, token_id + len(token))
        token_id += len(token)
    return tokens, word2token_map


def concat_two_sentences(tokenized_text1, tokenized_text2,
                         word2token_map1, word2token_map2,
                         evidence_word_span1, evidence_word_span2,
                         tokenizer):
    # connect two sentence by [CLS] text1 [SEP] text2 [SEP], for NLI task
    # tokenized_text: [token1, token2, ...]
    # word2token_map: {word id -> token span}
    # evidence_word_span: [(start word id, end word id), ...]
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    concated_tokenized_text = [cls_token_id] + tokenized_text1 + [sep_token_id] + tokenized_text2 + [sep_token_id]
    special_tokens_mask = [1] + [0] * len(tokenized_text1) + [1] + [0] * len(tokenized_text2) + [1]
    token_type_ids = [0] * (len(tokenized_text1) + 2) + [1] * (len(tokenized_text2) + 1)
    concated_word2token_map = {}  # sentence id -> word id -> token span
    for word_id, token_span in word2token_map1.items():
        concated_word2token_map[0, word_id] = (token_span[0] + 1, token_span[1] + 1)
    for word_id, token_span in word2token_map2.items():
        concated_word2token_map[1, word_id] = (token_span[0] + len(tokenized_text1) + 2, token_span[1] + len(tokenized_text1) + 2)
    concated_evidence_token_span = []
    for evidence_span in evidence_word_span1:
        start_token_id = concated_word2token_map[0, evidence_span[0]][0]
        end_token_id = concated_word2token_map[0, evidence_span[1]-1][1]
        concated_evidence_token_span.append((start_token_id, end_token_id))
    for evidence_span in evidence_word_span2:
        start_token_id = concated_word2token_map[1, evidence_span[0]][0]
        end_token_id = concated_word2token_map[1, evidence_span[1]-1][1]
        concated_evidence_token_span.append((start_token_id, end_token_id))

    evidence_token_mask = []  # evidence token is 1, others are 0
    for i in range(len(concated_tokenized_text)):
        if any([start_token_id <= i < end_token_id for start_token_id, end_token_id in concated_evidence_token_span]):
            evidence_token_mask.append(1)
        else:
            evidence_token_mask.append(0)

    return concated_tokenized_text, token_type_ids, special_tokens_mask, evidence_token_mask


def esnli_example_map(example, tokenizer, label_names):
    doc_id = example["doc_id"]
    text1 = example["premise"]
    text2 = example["hypothesis"]

    evidence_word_span1 = example["premise_evidence_span"]
    evidence_word_span2 = example["hypothesis_evidence_span"]
    label = example["classification"]
    tokenized_text1, word2token_map1 = get_tokenized_sentence(text1, tokenizer)
    tokenized_text2, word2token_map2 = get_tokenized_sentence(text2, tokenizer)
    concated_tokenized_text, token_type_ids, special_tokens_mask, evidence_token_mask = \
        concat_two_sentences(tokenized_text1, tokenized_text2,
                             word2token_map1, word2token_map2,
                             evidence_word_span1, evidence_word_span2, tokenizer)

    seq_length = len(concated_tokenized_text)
    attention_mask = [1] * len(concated_tokenized_text)

    return {
        "id": doc_id,
        "input_ids": concated_tokenized_text,
        "seq_length": seq_length,
        "token_type_ids": token_type_ids,
        "special_tokens_mask": special_tokens_mask,
        "attention_mask": attention_mask,
        "evidence_token_mask": evidence_token_mask,
        "label": label_names.index(label),
    }


def esnli_double_sentence_data_collator(features, tokenizer):
    # convert to tensors
    first = features[0]
    batch = {}
    for key in first.keys():
        if key in ["id", "seq_length"]:
            batch[key] = [example[key] for example in features]
        elif key in ["input_ids", "token_type_ids", "special_tokens_mask", "attention_mask", "evidence_token_mask"]:
            batch[key] = pad_sequence([torch.tensor(example[key], dtype=torch.long) for example in features], batch_first=True)
        if key == "label":
            batch["labels"] = torch.tensor([example[key] for example in features], dtype=torch.long).unsqueeze(-1)
    batch["token_word_position_map"] = get_token_word_position_map(batch, tokenizer)
    return batch

# END For processing eraser_esnli dataset

# For processing eraser_cose dataset


def cose_example_map(example, tokenizer, label_names):
    # TODO Add new dataset
    doc_id = example["doc_id"]


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

    elif config.data_set_name == "esnli":
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

    elif config.data_set_name == "cose":
        model_name_or_path = "bert-base-uncased"
        label_names = ["A", "B", "C", "D", "E"]
        problem_type = "single_label_classification"
        text_col_num = 2
        text_col_name = ["question", "query"]
        config.adapter_name = "AdapterHub/bert-base-uncased-pf-commonsense_qa"

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


def get_dataloader_and_model(config, dataset_config, tokenizer, return_simulate_dataloader=True, return_eval_dataloader=True):

    simulate_dataloader = None
    eval_dataloader = None
    model_name_or_path = dataset_config["model_name_or_path"]
    label_names = dataset_config["label_names"]
    num_labels = dataset_config["num_labels"]
    problem_type = dataset_config["problem_type"]
    text_col_num = dataset_config["text_col_num"]
    text_col_name = dataset_config["text_col_name"]

    if config.data_set_name in ["emotion"]:

        def add_id_and_label(example):
            example["id"] = hash(example["text"])
            example["label"] = label_names.index(example["label_name"])
            return example

        if return_simulate_dataloader or return_eval_dataloader:
            dataset = load_dataset('csv', data_files={'train': 'data/emotion/test.txt', 'eval': 'data/emotion/val.txt', 'test': 'data/emotion/test.txt'}, delimiter=";")
            dataset = dataset.map(add_id_and_label)
            train_dataset = dataset["train"]
            eval_dataset = dataset["eval"]
            data_collator = partial(single_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name=text_col_name)
            if return_simulate_dataloader:
                simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size)
            if return_eval_dataloader:
                eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    elif config.data_set_name in ["snli"]:
        if return_simulate_dataloader or return_eval_dataloader:
            dataset = load_dataset(config.data_set_name)
            label_dict = {0: 1, 1: 2, 2: 0}

            def add_id_and_fix_label(example):
                example["id"] = hash(example["premise"] + example["hypothesis"])
                example["label"] = label_dict[example["label"]]
                return example
            dataset = dataset.filter(lambda example: example['label'] != -1).map(add_id_and_fix_label)
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]

            data_collator = partial(double_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name1=text_col_name[0], text_col_name2=text_col_name[1])
            if return_simulate_dataloader:
                simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size)
            if return_eval_dataloader:
                eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["esnli"]:

        if return_simulate_dataloader or return_eval_dataloader:
            dataset = load_dataset("niurl/eraser_esnli")
            label_dict = {0: 1, 1: 2, 2: 0}

            def fix_label(example):
                example["label"] = label_dict[example["label"]]
                return example

            dataset = dataset.map(partial(esnli_example_map, tokenizer=tokenizer, label_names=label_names), num_proc=16).map(fix_label, num_proc=16)
            train_dataset = dataset["train"]
            eval_dataset = dataset["val"]

            data_collator = partial(esnli_double_sentence_data_collator, tokenizer=tokenizer)
            if return_simulate_dataloader:
                simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size)
            if return_eval_dataloader:
                eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["sst2"]:
        if return_simulate_dataloader or return_eval_dataloader:
            dataset = load_dataset("glue", config.data_set_name)
            dataset = dataset.rename_column("idx", "id")
            train_dataset = dataset["train"]
            eval_dataset = dataset["validation"]

            data_collator = partial(single_sentence_data_collator, tokenizer=tokenizer, num_labels=num_labels, problem_type=problem_type, text_col_name=text_col_name)
            if return_simulate_dataloader:
                simulate_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=config.batch_size)
            if return_eval_dataloader:
                eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=config.eval_test_batch_size)

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["cose"]:
        if return_simulate_dataloader or return_eval_dataloader:
            dataset = load_dataset("niurl/eraser_cose")
            dataset = dataset.map(partial(cose_example_map, tokenizer=tokenizer, label_names=label_names), num_proc=16)

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    print("teacher model loaded")
    if hasattr(teacher_model.config, "prediction_heads"): # for transformers >= 4.0.0
        assert teacher_model.config.prediction_heads[config.data_set_name]["num_labels"] == num_labels
    else: # for transformers < 4.0.0
        assert teacher_model.config.num_labels == num_labels

    return teacher_model, simulate_dataloader, eval_dataloader


def get_dataset_and_model(config, dataset_config, tokenizer):

    train_dataset = None
    eval_dataset = None
    model_name_or_path = dataset_config["model_name_or_path"]
    label_names = dataset_config["label_names"]
    num_labels = dataset_config["num_labels"]

    if config.data_set_name in ["emotion"]:
        dataset = load_dataset('csv', data_files={'train': 'data/emotion/test.txt', 'eval': 'data/emotion/val.txt', 'test': 'data/emotion/test.txt'}, delimiter=";")

        def add_id_and_label(example):
            example["id"] = hash(example["text"])
            example["label"] = label_names.index(example["label_name"])
            return example

        dataset = dataset.map(add_id_and_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["eval"]

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    elif config.data_set_name in ["snli"]:
        dataset = load_dataset(config.data_set_name)
        label_dict = {0: 1, 1: 2, 2: 0}

        def add_id_and_fix_label(example):
            example["id"] = hash(example["premise"] + example["hypothesis"])
            example["label"] = label_dict[example["label"]]
            return example
        dataset = dataset.filter(lambda example: example['label'] != -1).map(add_id_and_fix_label)
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["esnli"]:
        dataset = load_dataset("niurl/eraser_esnli")
        label_dict = {0: 1, 1: 2, 2: 0}

        def fix_label(example):
            example["label"] = label_dict[example["label"]]
            return example

        dataset = dataset.map(partial(esnli_example_map, tokenizer=tokenizer, label_names=label_names), num_proc=16).map(fix_label, num_proc=16)
        train_dataset = dataset["train"]
        eval_dataset = dataset["val"]

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["sst2"]:
        dataset = load_dataset("glue", config.data_set_name)
        dataset = dataset.rename_column("idx", "id")
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

        teacher_model = MyBertForSequenceClassification.from_pretrained(model_name_or_path)

    elif config.data_set_name in ["cose"]:
        dataset = load_dataset("niurl/eraser_cose")
        dataset = dataset.map(partial(cose_example_map, tokenizer=tokenizer, label_names=label_names), num_proc=16)

        from transformers import BertModelWithHeads
        teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
        adapter_name = teacher_model.load_adapter(config.adapter_name, source="hf")
        teacher_model.set_active_adapters(adapter_name)

    assert teacher_model.config.num_labels == num_labels

    return teacher_model, train_dataset, eval_dataset