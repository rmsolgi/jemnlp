
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import os
from evaluate import load
from src.applications.generation.gen_benchmarks import get_gen_bench
from src.network.utils import get_activations_empirical_cov
import pickle
from execute.setup.lib.eval import eval_ppl, eval_zero_shot
######################################################################################################

dataset_name="wikitext" #"alpaca"
SAVE_NAME = 'Llama_3_8b'
model_name = "meta-llama/Llama-2-7b-hf"
######################################################################################################

save_model_name=SAVE_NAME
saving_path=os.path.join('/data/solgi/Jruns', save_model_name)
os.makedirs(saving_path, exist_ok=True)

data_path=os.path.join('/data/solgi/Jruns', 'cov_data', SAVE_NAME+'_'+dataset_name)
os.makedirs(data_path, exist_ok=True)

######################################################################################################
######################################################################################################

model = AutoModelForCausalLM.from_pretrained(model_name, device_map = 'auto')
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced",  max_memory={
#         0: "5GiB", 
#         1: "5GiB",  # if you have more GPUs
#         2: "5GiB",
#         3: "5GiB",
#         "cpu": "200GiB"
#     })

# tokenizer = AutoTokenizer.from_pretrained(model_name)
from transformers import LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token  
tokenizer.padding_side = "left"  
print(model)
######################################################################################################
######################################################################################################

bench = get_gen_bench(tokenizer, dataset_name)
training_samples = bench.train_dataset.shuffle(seed=47).select(range(256))

######################################################################################################

act_emp_cov_dict=get_activations_empirical_cov(model, training_samples)
file_path=os.path.join(data_path, 'act_emp_cov_dict.pkl')
for key in act_emp_cov_dict:
    act_emp_cov_dict[key] = act_emp_cov_dict[key].cpu()
with open(file_path, "wb") as f:
    pickle.dump(act_emp_cov_dict, f)
# ######################################################################################################
# model.half() 
# model = model.to("cuda")
# model.seqlen = 4096
# ppl_test = eval_ppl(model, tokenizer, model.device)
# print(f"wikitext perplexity {ppl_test}")