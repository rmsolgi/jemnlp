
import torch
import os
import pickle
from src.layers.config import LayerConfig
from src.network.config import NetConfig
from src import layers
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm

def save_model_params(the_model, net_config_dict, dir):
    model_params_dict={}
    target_layers=net_config_dict['target_layers']
    for name, module in the_model.named_modules():
        if name in target_layers:
            # print(name)
            index = target_layers.index(name)
            key=net_config_dict['layer_configs'][index]['layer_name']
            model_params_dict[key] = module.get_params_dict()
    
    tm_params_path=os.path.join(dir,'tm_params.pth')
    torch.save(model_params_dict, tm_params_path)


def process_net_config(net_config_dict):
    processed_net_config_dict = {
                'target_layers': net_config_dict['target_layers'],
                'target_types': net_config_dict['target_types'],
                # 'layer_configs': layer_configs_list
            }
    layer_config_list = []
    for conf in net_config_dict['layer_configs']:
        conf['layer_class'] = getattr(layers, conf['layer_class_type']) 
        layer_config_list.append(LayerConfig(**conf))
    processed_net_config_dict['layer_configs'] = layer_config_list
    net_config=NetConfig(**processed_net_config_dict)
    return net_config



def get_activations_empirical_cov(model, dataset, batch_size=1, device="cuda"):
    model.eval()

    # def collate_fn(batch):
    #     return {
    #         k: torch.stack([d[k].clone().detach() for d in batch])
    #         for k in dataset.column_names
    #     }
    def collate_fn(batch):
        allowed_keys = {'input_ids', 'attention_mask', 'labels'}
        return {
            k: torch.stack([d[k].clone().detach() for d in batch])
            for k in batch[0]
            if k in allowed_keys
        }
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    activations = {}

    def hook_fn(module, input, output, layer_name):
        X = input[0]
        X = X.reshape(-1, X.shape[-1])
        XXT = X.T @ X
        if layer_name in activations:
            activations[layer_name] += XXT
        else:
            activations[layer_name] = XXT
    # def hook_fn(module, input, output, layer_name):
    #     X = input[0]
    #     X = X.reshape(-1, X.shape[-1])
        
    #     # Move a clone of X to CPU for matrix multiplication
    #     X_cpu = X.detach().to('cpu')
    #     XXT = X_cpu.T @ X_cpu  # Computed on CPU

    #     if layer_name in activations:
    #         activations[layer_name] += XXT
    #     else:
    #         activations[layer_name] = XXT
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(partial(hook_fn, layer_name=name)))

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing Cov"):
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(**batch)

    for hook in hooks:
        hook.remove()

    return activations

