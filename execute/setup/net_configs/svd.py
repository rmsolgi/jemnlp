import os
import pickle
def svd_llama_2_7b(ratio, info_path=None):

    target_layers_names=[]
    target_types_list=[]
    layer_configs_list=[]
    ######################################################################################
    mn = 4096*11008
    mpn = 4096+11008
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,32):
        layer_name='model.layers.'+str(i)+'.mlp.gate_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.up_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.down_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())


######################################################################################
    mn = 4096*4096
    mpn = 4096+4096
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,32):
        layer_name='model.layers.'+str(i)+'.self_attn.k_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.v_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.o_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.q_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

######################################################################################
    net_config_dict = {
                'target_layers': target_layers_names,
                'target_types': target_types_list,
                'layer_configs': layer_configs_list
            }

    return net_config_dict

######################################################################################
######################################################################################
######################################################################################

def svd_llm_llama_2_7b(ratio, info_path=None):
    cov_dict_path=os.path.join(info_path, 'act_emp_cov_dict.pkl')
    with open(cov_dict_path, "rb") as f:
            emp_cov_dict = pickle.load(f)

    net_config_dict = svd_llama_2_7b(ratio)
    for config in net_config_dict['layer_configs']:
        config['layer_class_type'] = 'SVD_LLM'
        config['info_path']=cov_dict_path
        config['covariance_matrix'] = emp_cov_dict[config['layer_name']]
    return net_config_dict

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def svd_llama_2_13b(ratio, info_path=None):

    target_layers_names=[]
    target_types_list=[]
    layer_configs_list=[]
    ######################################################################################
    mn = 5120*13824
    mpn = 5120+13824
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,40):
        layer_name='model.layers.'+str(i)+'.mlp.gate_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.up_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.down_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())


######################################################################################
    mn = 5120*5120
    mpn = 5120+5120
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,40):
        layer_name='model.layers.'+str(i)+'.self_attn.k_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.v_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.o_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.q_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

######################################################################################
    net_config_dict = {
                'target_layers': target_layers_names,
                'target_types': target_types_list,
                'layer_configs': layer_configs_list
            }

    return net_config_dict

######################################################################################
######################################################################################
######################################################################################
def svd_llm_llama_2_13b(ratio, info_path=None):
    cov_dict_path=os.path.join(info_path, 'act_emp_cov_dict.pkl')
    with open(cov_dict_path, "rb") as f:
            emp_cov_dict = pickle.load(f)

    net_config_dict = svd_llama_2_13b(ratio)
    for config in net_config_dict['layer_configs']:
        config['layer_class_type'] = 'SVD_LLM'
        config['info_path']=cov_dict_path
        config['covariance_matrix'] = emp_cov_dict[config['layer_name']]
    return net_config_dict

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################


def svd_mistral_7b(ratio, info_path=None):

    target_layers_names=[]
    target_types_list=[]
    layer_configs_list=[]
    ######################################################################################
    mn = 4096*14336
    mpn = 4096+14336
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,32):
        layer_name='model.layers.'+str(i)+'.mlp.gate_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.up_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.mlp.down_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())


######################################################################################
    mn = 4096*4096
    mpn = 4096+4096
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,32):


        layer_name='model.layers.'+str(i)+'.self_attn.o_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.q_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

######################################################################################
    mn = 4096*1024
    mpn = 4096+1024
    rank = int ((ratio * mn) / mpn)

    layer_config_dict={
            'layer_class_type': 'SVDW',
            'mode': 'teacher',
            'rank': rank
            }

    for i in range(0,32):
        layer_name='model.layers.'+str(i)+'.self_attn.k_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())

        layer_name='model.layers.'+str(i)+'.self_attn.v_proj'
        target_layers_names.append(layer_name)
        target_types_list.append('linear')
        layer_config_dict['layer_name']=layer_name
        layer_configs_list.append(layer_config_dict.copy())
#####################################################################################
    net_config_dict = {
                'target_layers': target_layers_names,
                'target_types': target_types_list,
                'layer_configs': layer_configs_list
            }

    return net_config_dict

######################################################################################
######################################################################################
######################################################################################

def svd_llm_mistral_7b(ratio, info_path=None):
    cov_dict_path=os.path.join(info_path, 'act_emp_cov_dict.pkl')
    with open(cov_dict_path, "rb") as f:
            emp_cov_dict = pickle.load(f)

    net_config_dict = svd_mistral_7b(ratio)
    for config in net_config_dict['layer_configs']:
        config['layer_class_type'] = 'SVD_LLM'
        config['info_path']=cov_dict_path
        config['covariance_matrix'] = emp_cov_dict[config['layer_name']]
    return net_config_dict

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################