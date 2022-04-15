# %%
from data_utils import *
from transformers import AutoTokenizer, set_seed
from accelerate.utils import send_to_device


model_name_or_path = "uer/roberta-base-finetuned-dianping-chinese"
label_names = ["negative", "positive"]
problem_type = "single_label_classification"
text_col_num = 1
text_col_name = "context"
token_quantity_correction = 2

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
MASK_TOKEN_ID = tokenizer.mask_token_id

from transformers import AutoModelForSequenceClassification
teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
teacher_model = send_to_device(teacher_model, device="cuda")

# %%
def single_sentence_data_collator(features, tokenizer, num_labels, problem_type, text_col_name="text", add_cls=True):

    first = features[0]
    batch = {}

    batch = tokenizer.batch_encode_plus([item[text_col_name] for item in features], add_special_tokens=add_cls,
                                        truncation=True, padding=True, max_length=256, return_special_tokens_mask=True, return_tensors='pt')

    batch["id"] = [item["id"] for item in features]
    batch["context"] = [item["context"] for item in features]
    batch["sent_token"] = [item["sent_token"] for item in features]

    return batch

dataset = load_dataset("json", data_files={"test":"data/du/data-part-1/senti_ch_part1.txt"})
train_dataset = None
eval_dataset = dataset["test"]
data_collator = partial(single_sentence_data_collator, tokenizer=tokenizer, num_labels=2, problem_type="", text_col_name="context")
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=32)


data = []
for simulate_step, simulate_batch in enumerate(eval_dataloader):
    ids = simulate_batch.pop("id")
    contexts = simulate_batch.pop("context")
    sent_tokens = simulate_batch.pop("sent_token")
    special_tokens_mask = simulate_batch.pop("special_tokens_mask")
    simulate_batch = send_to_device(simulate_batch, device="cuda")

    model_output = teacher_model(**simulate_batch, output_attentions=True)
    y_pred = model_output.logits.detach().argmax(1)
    y_pred = y_pred.cpu().tolist()
    
    for id, context, sent_token, label in zip(ids, contexts, sent_tokens, y_pred):
        # data.append({"id": id, "context": context, "sent_token": sent_token, "label": label})
        data.append({"id": id, "context": context, "label": label})


# save data to file
with open("data/du/data-part-1/senti_ch_part1_pred.txt", "w") as f:
    for item in data:
        f.write(json.dumps(item,ensure_ascii=False) + "\n")

# %%
