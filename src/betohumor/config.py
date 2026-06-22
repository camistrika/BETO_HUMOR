from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class DataConfig:
    seed: int = 42
    data_path: str = "data/raw/haha_2019_train.csv"
    processed_dir: str = "data/processed"
    text_col: str = "text"
    label_col: str = "is_humor"
    val_size: float = 0.1
    max_length: int = 128


@dataclass
class BetoConfig:
    base_model: str = "dccuchile/bert-base-spanish-wwm-cased"
    num_labels: int = 2
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    early_stopping_patience: int = 3
    output_dir: str = "results/checkpoints"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["query", "value"])


@dataclass
class NbsvmConfig:
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 3
    max_df: float = 0.9
    C: float = 4.0
