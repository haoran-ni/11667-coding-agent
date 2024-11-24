import argparse
import os
from dataclasses import dataclass
from typing import Literal

import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, \
    BitsAndBytesConfig
from trl import SFTTrainer


@dataclass
class ModelConfig:
    model_name: str
    dataset_name: str
    dataset_split: str


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


def main():
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config
    os.environ["WANDB_PROJECT"] = "11667-llms-hw6"

    # Parse configuration arguments
    hfparser = HfArgumentParser((ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments))
    model_config, custom_bnb_config, custom_lora_config, sft_config, train_args = hfparser.parse_json_file(
        json_file=config_path)  # type: ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments

    # Retrieve dataset
    full_dataset = load_dataset(model_config.dataset_name, split=model_config.dataset_split)

    # Split dataset deterministically (e.g., 95% train, 5% validation)
    train_test_split = full_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)

    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']  # use as evaluation/test dataset

    # Format the instruction using a template
    def format_instruction(sample):
        output_texts = []
        for i in range(len(sample['instruction'])):
            text = f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a programming code that can solve the following Task:

### Task:
{sample['instruction'][i]}

### Input:
{sample['input'][i]}

### Response:
{sample['output'][i]}
            """
            output_texts.append(text)
        return output_texts

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=custom_bnb_config.load_in_4bit,
        bnb_4bit_quant_type=custom_bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, custom_bnb_config.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=custom_bnb_config.bnb_4bit_use_double_quant
    )

    # Retrieve model
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name, quantization_config=bnb_config,
                                                 use_cache=False, device_map={"": 0})

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=custom_lora_config.lora_r,
        lora_alpha=custom_lora_config.lora_alpha,
        lora_dropout=custom_lora_config.lora_dropout,
        bias=custom_lora_config.lora_bias,
        task_type=custom_lora_config.lora_task_type,
    )

    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        max_seq_length=sft_config.sft_max_seq_length,
        tokenizer=tokenizer,
        packing=sft_config.sft_packing,
        formatting_func=format_instruction,
        args=train_args,
    )

    # Train
    print("Start Training")
    trainer.train()
    print("End Training")

    # Save model
    trainer.save_model()
    print("Model saved")

    # Empty VRAM to free up resources
    del model
    del trainer
    import gc
    gc.collect()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    # Load the saved model
    model = AutoPeftModelForCausalLM.from_pretrained(
        train_args.output_dir,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(f"{train_args.output_dir}/merged_model", safe_serialization=True)
    tokenizer.save_pretrained(f"{train_args.output_dir}/merged_model")


if __name__ == "__main__":
    main()
