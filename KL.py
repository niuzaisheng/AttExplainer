
# %%
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from scipy.special import kl_div

# 加载GPT-2模型和分词器
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 自定义函数以获取生成文本的隐藏状态
def get_hidden_states_during_generation(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    max_length = 50  # 您可以设置自己的最大长度
    generated_ids = model.generate(input_ids, max_length=max_length)
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    print(100 * '-')
    with torch.no_grad():
        outputs = model(input_ids=generated_ids, output_hidden_states=True)
    last_hidden_states = outputs.hidden_states[-1]  # 取最后一层的隐藏状态
    summed_last_hidden_states = torch.sum(last_hidden_states, dim=1)  # 在sequence length维度上求和
    return summed_last_hidden_states

# 原始prompt
prompt_orig = "Once upon a time"
e_orig = get_hidden_states_during_generation(prompt_orig).flatten()

# 微扰后的prompt
prompt_pertur = "Long ago and far away"
e_pertur = get_hidden_states_during_generation(prompt_pertur).flatten()

# %%
# 计算两者之间的KL散度
e_orig_normalized = torch.nn.functional.softmax(e_orig)
e_pertur_normalized = torch.nn.functional.softmax(e_pertur)

# %%
# kl_divergence = kl_div(e_orig_normalized.numpy(), e_pertur_normalized.numpy())
kl_divergence = kl_div(e_orig.numpy(), e_pertur.numpy())
print("KL Divergence: " + str(kl_divergence))
# %%
