from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EnvConfig:
    SEED: int = 42
    CUDA_DEVICE_ORDER: str = "PCI_BUS_ID"
    OMP_NUM_THREADS: str = "4"
    CUDA_VISIBLE_DEVICES: str  = "1"
    TRANSFORMERS_CACHE: str = "/home/dev/public-datasets/e.shvetsov/hugging_models"

@dataclass
class PathsConfig:
    data_dir: str = "data/"
    log_dir: str = "artifacts/logs/"
    checkpoint_dir: str = "artifacts/checkpoints/"
    results_dir: str = "artifacts/results/"

@dataclass
class ModelConfig:
    path: str = "/home/dev/public-datasets/e.shvetsov/hugging_models/Llama-3.1-8B-Instruct"
    seqlen: int = 512

@dataclass
class PPLWikitext2Config:
    run_ppl: bool = False
    batch_size: int = 8

@dataclass
class HarnessConfig:
    run_lm_eval: bool = True
    tasks: List[str] = field(default_factory=lambda: ["arc_challenge", "boolq", "arc_easy", "piqa", "winogrande", "hellaswag"])
    num_fewshot: int = 0
    batch_size: int = 512
    apply_chat_template: bool = False

@dataclass
class BenchmarksConfig:
    ppl_wikitext2: PPLWikitext2Config = field(default_factory=PPLWikitext2Config)
    harness: HarnessConfig = field(default_factory=HarnessConfig)

@dataclass
class PruningConfig:
    sparsity_type: str = "semi-structured_act_magnitude"
    transformation_type: str = "variance"
    sparsity_ratio: float = 0.5
    additional_transformation: str = "none"
    prune_n: int = 2
    prune_m: int = 4
    module: str = "layers"
    target_modules: List[str] = field(default_factory=lambda: ["q_proj","k_proj","v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class FinetuningConfig:
    type: str = "global"
    output_dir: str = "/home/LLM_activation_pruning/act_prune/artifacts/models/llama2_7b_wiki"
    dataset_name: str = "Salesforce/wikitext"
    dataset_config_name: str = "wikitext-2-raw-v1"
    seed: int = 11
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 8
    trust_remote_code: bool = True
    num_train_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.03
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = False
    save_strategy: str = "steps"
    save_steps: int = 200
    report_to: Optional[str] = None

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    benchmarks: BenchmarksConfig = field(default_factory=BenchmarksConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    finetuning: FinetuningConfig = field(default_factory=FinetuningConfig)