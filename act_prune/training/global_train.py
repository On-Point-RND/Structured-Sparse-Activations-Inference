from argparse import ArgumentParser
import logging
import math
import os
import random
import shutil
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from itertools import chain
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments
)
from peft import (
    get_peft_model,
    TaskType,
    LoraConfig
)

IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    max_memory: int = field(
        default=21,
        metadata={"help": "Free memory per gpu."}
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    quant_noise_config: dict = field(
        default=None,
        metadata={"help": "Parameters to add noise"},
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


def run_train(
    model,
    tokenizer,
    config
):
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
    
    # Load pretrained tokenizer
    # tokenizer_kwargs = {
    #     "cache_dir": model_args.cache_dir,
    #     "use_fast": model_args.use_fast_tokenizer,
    #     "revision": model_args.model_revision,
    #     "token": model_args.token,
    #     "trust_remote_code": model_args.trust_remote_code,
    # }

    # if model_args.tokenizer_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    # elif model_args.model_name_or_path:
    #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    # Load pretrained model
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     torch_dtype=torch.bfloat16,
    #     token=model_args.token,
    #     device_map = 'auto'
    # )

    # for name, param in model.named_parameters():
    #     param.requires_grad = False




    #Load and preprocessing dataset
    tokenizer.pad_token = tokenizer.eos_token
    raw_datasets = load_hf_datasets(data_args)
    tokenized_datasets = tokenize_datasets(data_args, raw_datasets, tokenizer)
    lm_datasets = format_datasets(data_args, tokenized_datasets, tokenizer)

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets.get("validation", None)

    logging.info("dataset prepared")

    # data_collator = DataCollatorWithMaskForCausalLM(
    #     tokenizer=tokenizer
    # )

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logging.info(f"trainable_params: {trainable_params}")

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    )

    train_result = trainer.train()

    # trainer.save_model()  # Saves the tokenizer too for easy upload
    return model