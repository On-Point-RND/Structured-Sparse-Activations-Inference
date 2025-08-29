import torch
from modelling.layers.linear_act_sp import Linear_act_sp
from modelling.blocks.mlp_act_sp import MLP_act_sp
from modelling.blocks.llama_attn import LlamaAttention_act_sp

def swap_linear_inplace(root_module, 
                sparsity_type,
                sparsity_ratio,
                prune_n,
                prune_m,
                target_layers,
                logging):
    
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
            if isinstance(module, torch.nn.Linear):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]
                
                if name[(ind+1):] in target_layers:            
                    sparse_linear = Linear_act_sp.from_original(
                        module, 
                        sparsity_type=sparsity_type,
                        sparsity_ratio=sparsity_ratio,
                        prune_n=prune_n,
                        prune_m=prune_m,  
                        name=name[(ind+1):]
                    )
                    setattr(father, name[ind + 1 :], sparse_linear)
                    logging.info(name)

    
def swap_mlp_inplace(root_module, 
                     orig_mlp_block,
                     sparsity_type,
                     logging):
     
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
            if isinstance(module, type(orig_mlp_block)):
                ind = name.rfind(".")
                if ind == -1:
                    father = module_name_dict[""]
                else:
                    father = module_name_dict[name[:ind]]
                
                sparse_mlp = MLP_act_sp.from_original(module, sparsity_type=sparsity_type)
                setattr(father, name[ind + 1 :], sparse_mlp)
                logging.info(name)

def swap_attention_inplace(root_module, 
                           orig_self_attn_block,
                           sparsity_type,
                           logging):
      
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, type(orig_self_attn_block)):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            sp_SelfAttn = LlamaAttention_act_sp.from_original(module, sparsity_type=sparsity_type)
            setattr(father, name[ind + 1 :], sp_SelfAttn)
            logging.info(name)
    