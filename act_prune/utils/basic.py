import os
import yaml
import random
from typing import Dict

import torch
import numpy as np


def load_yaml(file_path: str) -> Dict:
    """Load a single YAML file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
    

def seed_everything(seed: int) -> None:
    """Fix seed everywhere."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic=True

def load_config(base_config_path: str, config_dir: str) -> Dict:
    """Load base configuration and merge sub-configurations."""
    base_config = load_yaml(base_config_path)
    return base_config