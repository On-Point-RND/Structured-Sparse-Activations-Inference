from datasets import load_dataset
import torch
import torch.nn as nn
from torch.amp import autocast
from transformers.utils import logging
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import math
import time
import os
import pandas as pd
from utils.getwiki import get_wikitext2
from tqdm import tqdm

from modelling.layers.linear_act_sp import Linear_act_sp

logger = logging.get_logger(__name__)


class TensorData(torch.utils.data.Dataset):
    def __init__(self, data, targets, device):
        self.data = data
        self.targets = targets
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x.to(self.device), y.to(self.device)

    def __len__(self):
        return len(self.targets)    

class TensorData_infer(torch.utils.data.Dataset):
    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __getitem__(self, index):
        x = self.data[index]
        return x.to(self.device)

    def __len__(self):
        return len(self.data)    

class TensorDataLoader:
    def __init__(self, dataset, batch_size, shuffle, num_workers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def get_loader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=False
        )


def find_layers(module, layers=[nn.Linear, Linear_act_sp], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def prepare_calibration_input(args, model, tokenizer, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype

    inps = torch.zeros((args.nsamples, args.seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    inps.requires_grad = False
    i = 0
    for batch in tqdm(dataloader):
        try:
            cache = model.model(batch[0].to(device),flag=True)
            inps[i] = cache[0].to("cpu")
            i += 1
        except ValueError:
            pass 
    outs = torch.zeros_like(inps)
    attention_mask = (batch[0] != tokenizer.pad_token_id).to(device)
    position_ids = torch.arange(batch[0].shape[1], device=device).unsqueeze(0)
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 


def val(layer, inps, outs, dataloader, batch_size, device, attention_mask, position_embeddings, layer_index=None):
    ret_loss = 0
    len_dataloader = len(dataloader)
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, batch_size, shuffle=False, num_workers=0).get_loader()
    criterion = nn.MSELoss(reduction="mean").to(device)
    with torch.no_grad():
        for inputs, outs in tensordata_loader:
            outputs = layer(inputs, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            loss = criterion(outputs, outs)

            ret_loss += (loss.detach().cpu().item()) * len(inputs)
    return ret_loss / len(inps)


def mark_only_shift_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'shift' not in n:
            p.requires_grad = False


def mark_only_bias_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'bias' not in n:
            p.requires_grad = False


def mark_only_scale_and_shift_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'scale' not in n and 'shift' not in n:
            p.requires_grad = False


def mark_only_lora_AB_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        print(n)
        if 'low_rank_A' not in n and 'low_rank_B' not in n:
            p.requires_grad = False


def prepare_optimizer_and_scheduler(layer, config, max_steps):
    weight_decay = config["finetuning"]["weight_decay"]
    learning_rate = config["finetuning"]["learning_rate"]
    adam_beta1 = config["finetuning"].get("adam_beta1", 0.9)
    adam_beta2 = config["finetuning"].get("adam_beta2", 0.95)
    learnable_params = config["finetuning"].get("parameters", None)
    warmup_steps = config["finetuning"]["warmup_ratio"] * max_steps
    warmup_steps = int(warmup_steps)


    def log_params(param_groups, des):
        for i, grouped_parameters in enumerate(param_groups):
            logger.info(
                f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay:{grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

    model_params = []
    if learnable_params in ["shift", "shift2", "var_shift"]:
        model_params = [p for n, p in layer.named_parameters() if 'shift' in n]
    elif learnable_params in ["bias1", "bias2"]:
        model_params = [p for n, p in layer.named_parameters() if 'bias' in n]
    elif learnable_params in ["scale_shift", "var_scale_shift"]:
        model_params = [p for n, p in layer.named_parameters() if 'shift' in n or 'scale' in n]
    elif learnable_params == 'r-sparce':
        model_params = [p for n, p in layer.named_parameters() if 'low_rank_A' in n or 'low_rank_B' in n]

    optimizer = AdamW(
        model_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(adam_beta1, adam_beta2),
    )
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps
    )
    
    return optimizer, lr_scheduler


def train(layer, inps, outs, dataloader, config, device, attention_mask, position_embeddings, layer_index=None):
    batch_size = config["finetuning"]["per_device_train_batch_size"]
    num_train_epoch = config["finetuning"]["num_train_epochs"]
    max_grad_norm = config["finetuning"].get("max_grad_norm", 1.0)
    learnable_params = config["finetuning"].get("parameters", None)
    
    init_loss = val(layer, inps, outs, dataloader, batch_size, device, attention_mask, position_embeddings, layer_index=layer_index)

    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // batch_size
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    max_steps = math.ceil(num_train_epoch * num_update_steps_per_epoch)

    if learnable_params in ["shift", "shift2", "var_shift"]:
        mark_only_shift_as_trainable(layer)
    elif learnable_params in ["bias1", "bias2"]:
        mark_only_bias_as_trainable(layer)    
    elif learnable_params in ["scale_shift", "var_scale_shift"]:
        mark_only_scale_and_shift_as_trainable(layer)    
    elif learnable_params == 'r-sparce':
        mark_only_lora_AB_as_trainable(layer)
    
    optimizer, lr_scheduler = prepare_optimizer_and_scheduler(layer, config, max_steps)
    criterion = nn.MSELoss(reduction="mean").cuda()
    losses = []
    lrs = []
    start_time = time.time()
    tensordata = TensorData(inps, outs, device)
    tensordata_loader = TensorDataLoader(tensordata, batch_size, shuffle=True, num_workers=0).get_loader()
    for epoch in range(0, num_train_epoch):
        print("epoch {}".format(epoch))
        for inputs, outps in tensordata_loader:
            outputs = layer(inputs, attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            loss = criterion(outputs, outps)
            lr = lr_scheduler.get_last_lr()[0]
            lrs.append(lr)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                        layer.parameters(), max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            layer.zero_grad()

            losses.append(loss.detach().cpu().item())

    torch.cuda.empty_cache()
    end_time = time.time()
    print("time cost of finetuning each layer: {}".format(end_time-start_time))

    final_loss = val(layer, inps, outs, dataloader, batch_size, device,attention_mask, position_embeddings, layer_index=layer_index)
    
    print(init_loss)
    print("*********")
    print(final_loss)
    
    return init_loss, final_loss


def sequential_parameter_training(config, model, dataloader, dev=torch.device("cuda:0")):
    print("Starting...")
    nsamples = len(dataloader)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_embeddings"] = kwargs["position_embeddings"]
            raise ValueError    

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    inps = inps.detach()
    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_embeddings = cache["position_embeddings"]

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        for name in subset:
            subset[name].sparsity_type = None

        with torch.no_grad():
            for j in range(0, nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), 
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings
                )[0]
                
        for name in subset:
            subset[name].sparsity_type = config["pruning"]["sparsity_type"]

        init_loss, final_loss = train(
            layer, inps, outs, dataloader, config, 
            dev, attention_mask=attention_mask, 
            position_embeddings=position_embeddings, 
            layer_index=i
        )

        with torch.no_grad():
            for j in range(0, nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0), 
                    attention_mask=attention_mask, 
                    position_embeddings=position_embeddings
                )[0]    

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    model = model.to(device=dev)