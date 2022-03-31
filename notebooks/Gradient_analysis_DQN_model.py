
# %%
# Gradient analysis DQN model

import torch
from torch import nn
from torch.nn import functional as F

# %%

class DQNNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.bins_num = 32
        self.matrix_num = 1

        self.h1 = self.matrix_num * (self.bins_num * 2 + 4)
        self.h2 = 63
        self.h3 = 8
        self.layer1 = nn.Linear(self.h1, self.h2)
        self.layer2 = nn.Linear(self.h2 + 1, self.h3)
        self.out = nn.Linear(self.h3, 1)

    def forward(self, x, s, seq_len):
        batch_size, matrix_num, max_seq_len, _ = x.size()

        x = x.reshape(batch_size, matrix_num, max_seq_len, self.bins_num * 2 + 4)  # [B, matrix_num, seq, (bins_num * 2 + 4)]
        x = x.permute(0, 2, 3, 1)  # [Batch x seq x (bins_num+2) x matrix_num]
        x = x.reshape(batch_size, max_seq_len, self.h1)  # [B, matrix_num, seq, 8]
        x = self.layer1(x)  # [B, seq, bins_num x 4]
        x = torch.relu(x)
        x = F.dropout(x)
        x = torch.cat([x, s.unsqueeze(-1)], dim=-1)  # [B, seq, h2 + 1]
        x = self.layer2(x)
        x = torch.relu(x)
        x = F.dropout(x)
        x = self.out(x)
        return x.reshape(batch_size, max_seq_len)  # [B , seq]

old_model = DQNNet()
old_model.load_state_dict(torch.load("report_weights/emotion_attacker_1M.bin",map_location="cpu"))

# %%
from transformers import BertModelWithHeads,AutoTokenizer
from utils import get_attention_features
import matplotlib.pyplot as plt
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
teacher_model = BertModelWithHeads.from_pretrained("bert-base-uncased")
adapter_name = teacher_model.load_adapter("AdapterHub/bert-base-uncased-pf-emotion", source="hf")
teacher_model.set_active_adapters(adapter_name)

# %%

inputs = tokenizer(["Hello, my dog is cute."], return_tensors="pt")
inputs_len = inputs["input_ids"].size(-1)
with torch.no_grad():
    model_outputs = teacher_model(**inputs, output_attentions=True)

    batch_attentions = model_outputs.attentions
    attentions = torch.cat([torch.mean(layer, dim=1, keepdim=True) for layer in batch_attentions], dim=1)
    raw_att = attentions.mean(1).detach()[0]

    att = get_attention_features(model_outputs, attention_mask=inputs["attention_mask"] , batch_seq_len=[inputs_len], bins_num=32).unsqueeze(1).detach()

att = att.clone()
att.requires_grad=True
s = torch.ones((1,inputs_len),requires_grad=True)

out = old_model(att, s, None)
out.backward(torch.ones(out.size()))

# %%
value_map = att.grad[0,0].numpy()
att_feature = att[0,0].detach().numpy()

fig, (ax, bx, cx) = plt.subplots(1,3, figsize=(10, 5))
ax.imshow(raw_att)
ax.invert_yaxis()

bx.imshow(att_feature.T, aspect='auto')
bx.invert_yaxis()

cx.imshow(value_map.T, aspect='auto')
cx.invert_yaxis()
plt.show()

# %%
