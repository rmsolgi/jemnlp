
from src.network.remodel import swap_layers
from transformers import  AutoModelForCausalLM
import os
import pickle
import torch
import time
from execute.setup import net_configs
from execute.setup.lib.eval import eval_ppl, eval_zero_shot
from transformers import LlamaTokenizer
from src.network.utils import save_model_params
############################################################################################
DO_COMPRESS = False
EVAL = True
SAVE = True
############################################################################################
# BASE_MODEL_NAME = 'llama_2_13b'
DATA_DIR = '/data/jemnlp'
COMPRESSION_CONFIG = 'svd_llm_mistral_7b'
TEMP_DIR = DATA_DIR
COMPRESSED_DIR = DATA_DIR
model_name = "mistralai/Mistral-7B-v0.1"
COV_NAME = 'mistral_7b_v01_wikitext'
ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
for ratio in ratios:
    print('ratio:', ratio)
    ############################################################################################
    ############################################################################################
    
    init_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    total_params = sum(p.numel() for p in init_model.parameters())
    print(f"Total parameters of the base: {total_params:,}")

    ###########################################################################################
    compressed_path=os.path.join(COMPRESSED_DIR, 'params', COMPRESSION_CONFIG+'_'+str(int(ratio*100)))
    compressed_model_dir = os.path.join(COMPRESSED_DIR,'checkpoints', COMPRESSION_CONFIG+'_'+str(int(ratio*100)))
    ###########################################################################################
    #############################################################################################
    get_config = getattr(net_configs, COMPRESSION_CONFIG) 
    info_path = os.path.join(DATA_DIR, 'cov_data', COV_NAME)
    net_config=get_config(ratio, info_path)
    #############################################################################################
    #############################################################################################
    if DO_COMPRESS:
        os.makedirs(compressed_path, exist_ok=True)
        start = time.time()
        new_model=swap_layers(init_model,net_config)
        end = time.time()
        print(f"Compression time: {end - start:.4f} seconds")
        save_model_params(new_model, net_config, compressed_path)
    else:
        params_dict=torch.load(os.path.join(compressed_path, 'tm_params.pth'))
        new_model=swap_layers(init_model,net_config, params_dict)
    #############################################################################################
    total_params = sum(p.numel() for p in new_model.parameters())
    print(f"Total parameters of compressed: {total_params:,}")
    #############################################################################################
    if EVAL: 
        new_model.half() 
        new_model = new_model.to("cuda")
        new_model.seqlen = 4096
        ppl_test = eval_ppl(new_model, tokenizer, new_model.device)
        print(f"wikitext perplexity {ppl_test}")
    if SAVE:
        new_model = new_model.half()
        os.makedirs(compressed_model_dir, exist_ok=True)
        compressed_model_save_path = os.path.join(compressed_model_dir, 'model.pt')
        torch.save(new_model, compressed_model_save_path)