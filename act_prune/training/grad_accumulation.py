from datasets import load_dataset
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.amp import GradScaler
from transformers.utils import logging
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
import math
import time
import os
import pandas as pd
from utils.getwiki import get_wikitext2
from tqdm import tqdm

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
from itertools import chain

from modelling.layers.linear_act_sp import Linear_act_sp

scaler = GradScaler()
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom dataset defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    dataset_percentage: Optional[int] = field(
        default=100,
        metadata={
            "help": "The percentage of the dataset used for computation"
        },  
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
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
        #print(child)
        if name1.find("lm_head") == -1:
            res.update(find_layers(
                child, layers=layers, name=name + '.' + name1 if name != '' else name1
            ))
    return res

def load_hf_datasets(
    data_args
):
    # Load the dataset
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            streaming=data_args.streaming,
            trust_remote_code=data_args.trust_remote_code
        )

        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         streaming=data_args.streaming,
        #         trust_remote_code=data_args.trust_remote_code
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         streaming=data_args.streaming,
        #         trust_remote_code=data_args.trust_remote_code
        #     )
        
        # if data_args.dataset_percentage < 100:
        #     dataset_frac = data_args.dataset_percentage/100
        #     dataset_parts = raw_datasets['train'].train_test_split(train_size=dataset_frac)
        #     raw_datasets['train'] = dataset_parts['train']
        #     dataset_parts = raw_datasets['validation'].train_test_split(test_size=dataset_frac)
        #     raw_datasets['validation'] = dataset_parts['test']

        return raw_datasets

def tokenize_datasets(
    data_args,
    raw_datasets,
    tokenizer
):
    
    dataset_type = list(raw_datasets.keys())[0]
    column_names = list(raw_datasets[dataset_type].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        output = tokenizer(examples[text_column_name])
        return output
    
    if not data_args.streaming:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    else:
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    return tokenized_datasets


def format_datasets(
    data_args,
    tokenized_datasets,
    tokenizer
):
    
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    print(max_seq_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(
        examples
    ):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        
        return result


    if not data_args.streaming:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {max_seq_length}",
        )
    else:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
        )
    
    return lm_datasets


def compute_grads(model, tokenizer, config, dev=torch.device("cuda:0")):
    config_ft = config["finetuning"]

    data_args = DataTrainingArguments(
        dataset_name = config_ft['dataset_name'],
        dataset_config_name = config_ft['dataset_config_name'],
        # validation_split_percentage = config_ft['validation_split_percentage'],
        max_seq_length = config_ft['max_seq_length'],
        # dataset_percentage = config_ft['dataset_percentage'],
        trust_remote_code = config_ft['trust_remote_code'],
        preprocessing_num_workers = config_ft['preprocessing_num_workers']
    )

    training_args = TrainingArguments(
        # run_name=config.run_name,
        output_dir = config_ft['output_dir'],
        overwrite_output_dir = True,
        learning_rate = config_ft['learning_rate'], 
        seed = config_ft['seed'], 
        # max_steps = config_ft['max_steps'],
        num_train_epochs = config_ft['num_train_epochs'], #3,
        weight_decay = config_ft['weight_decay'], #0.1,
        warmup_ratio = config_ft['warmup_ratio'],
        lr_scheduler_type = config_ft['lr_scheduler_type'],
        per_device_train_batch_size = config_ft['per_device_train_batch_size'], #2,
        per_device_eval_batch_size = config_ft['per_device_eval_batch_size'], #2,
        gradient_accumulation_steps = config_ft['gradient_accumulation_steps'], #16,
        gradient_checkpointing=config_ft['gradient_checkpointing'], #False,
        save_strategy = config_ft['save_strategy'],
        save_steps = config_ft['save_steps'],
        # evaluation_strategy = config.evaluation_strategy,
        # eval_steps = config.eval_steps,
        bf16=True,
        logging_steps = 1,
        do_train = True,
        do_eval = False,
        # report_to = config['report_to']
    )

    use_cache = model.config.use_cache
    model.config.use_cache = False

    #Load and preprocessing dataset
    tokenizer.pad_token = tokenizer.eos_token
    raw_datasets = load_hf_datasets(data_args)
    tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
    lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation", None)

    logger.info("dataset prepared")

    subset = find_layers(model)
    for name in subset:
        subset[name].sparsity_type = None

    def add_grad(name):
        def tmp(module, grad_input, grad_output):
            subset[name].add_grad(grad_input, grad_output)
        return tmp

    handles = []
    for name in subset:
        handles.append(subset[name].register_full_backward_hook(add_grad(name)))

    bs = training_args.per_device_train_batch_size
    # nsamples = train_dataset.num_rows
    nsamples = 2

    for i in range(0, nsamples, bs):

        if i % 50 == 0:
            print(f"sample {i}")
        j = min(i + bs, nsamples)

        items = train_dataset[i:j]
        input_ids = torch.tensor(items['input_ids'], device=dev)
        labels = torch.tensor(items['labels'], device=dev)
        

        loss = model(
            input_ids=input_ids,
            labels=labels
        ).loss
        loss.backward()

        del input_ids
        del labels
        torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    for name in subset:
        subset[name].sparsity_type = config["pruning"]["sparsity_type"] 

    model.config.use_cache = use_cache