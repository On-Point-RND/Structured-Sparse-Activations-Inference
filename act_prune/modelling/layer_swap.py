import torch
from modelling.layers.linear_act_sp import Linear_act_sp
from modelling.blocks.mlp_act_sp import MLP_act_sp
from modelling.blocks.llama_attn import LlamaAttention_act_sp


def weight_prune(layer, 
                 sparsity_type, 
                 transformation_type,
                 sparsity_ratio: float, 
                 prune_n, 
                 prune_m, 
                 name):
    
    w = layer.weight.data
    b = layer.bias.data if layer.bias is not None else None

    if sparsity_type == "semi-structured_weight_magnitude":
        orig_shape = w.shape
        w_1d = w.view(-1, prune_m)

        _, idx = torch.topk(w_1d.abs(), prune_n, dim=1, sorted=False)
        mask_1d = torch.zeros_like(w_1d)
        mask_1d.scatter_(dim=1, index=idx, value=True)
        mask = mask_1d.view(orig_shape)
        pruned_w = w * mask

    elif sparsity_type == "unstructured_weight_magnitude":
        w_1d = w.abs().flatten()
        k = int(len(w_1d) * sparsity_ratio)
        thresh = torch.kthvalue(w_1d, k)[0]
        mask = w.abs() >= thresh
        pruned_w = w * mask
    else:
        raise ValueError(f"Unsupported sparsity type: {sparsity_type}")


    new_layer = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None)
    new_layer.weight.data = pruned_w

    if b is not None:
        new_layer.bias.data = b.clone()

    new_layer.name = name
    return new_layer

def swap_linear_inplace(root_module, 
                sparsity_type,
                transformation_type,
                sparsity_ratio,
                learnable_params,
                prune_n,
                prune_m,
                target_layers,
                logging):
    
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    replaced_cnt = 0
    for name, module in module_name_dict.items():
        if isinstance(module, torch.nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]

            if name[(ind+1):] in target_layers:
                kvargs = dict(
                    sparsity_type=sparsity_type,
                    transformation_type=transformation_type,
                    sparsity_ratio=sparsity_ratio,
                    prune_n=prune_n,
                    prune_m=prune_m,
                    name=name[(ind + 1) :],
                    learnable_params=learnable_params
                )
                if sparsity_type in ("semi-structured_act_magnitude",
                                     "unstructured_act_magnitude",
                                     "semi-structured_act_magnitude_var_weight", 
                                     "semi-structured_act_grad_acc",
                                     "semi_structural_L_pruner",
                                     "unstructured_clact_pruner",
                                     "unstructured_amber_pruner"):
                    
                    sparse_linear = Linear_act_sp.from_original(module, **kvargs)
                elif sparsity_type in ("semi-structured_weight_magnitude", "unstructured_weight_magnitude"):
                    sparse_linear = weight_prune(module, **kvargs)
                else:
                    raise ValueError(f"Unsupported sparsity type: {sparsity_type}")

                setattr(father, name[ind + 1 :], sparse_linear)
                replaced_cnt += 1
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
    