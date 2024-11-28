from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    model_name: str
    dataset_name: str
    dataset_split: str
    attn_implementation: str
    hp_tuning: bool


@dataclass
class CustomBnBConfig:
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool


@dataclass
class CustomLoRAConfig:
    lora_r: int
    # lora_target_modules: str
    lora_alpha: int
    lora_dropout: float
    lora_bias: Literal["none", "all", "lora_only"]
    lora_task_type: str


@dataclass
class SFTConfig:
    sft_max_seq_length: int
    sft_packing: bool


@dataclass
class EvaluationConfig:
    eval_on_validation: bool
    is_unmerged: bool
    is_bnb: bool
    ppl_only: bool
    legacy_generation: bool
