import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, \
    BitsAndBytesConfig
from trl import SFTTrainer

from configuration import ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig
from templates import format_user_text, format_assistant_text


def main():
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")  # Positional argument
    args = parser.parse_args()
    config_path = args.config
    os.environ["WANDB_PROJECT"] = "11667-llms-hw6-sft"

    # Parse configuration arguments
    hfparser = HfArgumentParser((ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments))
    model_config, custom_bnb_config, custom_lora_config, sft_config, train_args = hfparser.parse_json_file(
        json_file=config_path)  # type: ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments

    # Retrieve dataset
    full_dataset = load_dataset(model_config.dataset_name, split=model_config.dataset_split)

    # Split dataset into train (95%), validation (4%), and test (1%)
    train_val_split = full_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)

    val_test_split = train_val_split['test'].train_test_split(test_size=0.2, shuffle=True, seed=42)

    train_dataset = train_val_split['train']
    eval_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    if model_config.hp_tuning:
        train_dataset = train_dataset.shuffle(seed=42).select(range(2000))

    # Format the dataset to be used in the instruct model
    def format_instruct(sample):
        output_json = []
        for i in range(len(sample['instruction'])):
            user_text = format_user_text(sample['instruction'][i], sample['input'][i])
            assistant_text = format_assistant_text(sample['output'][i])
            row_json = [{"role": "user", "content": user_text}, {"role": "assistant", "content": assistant_text}]
            output_json.append(row_json)
        return tokenizer.apply_chat_template(output_json, tokenize=False)

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=custom_bnb_config.load_in_4bit,
        bnb_4bit_quant_type=custom_bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=custom_bnb_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=custom_bnb_config.bnb_4bit_use_double_quant
    )

    # Retrieve model
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name,
                                                 quantization_config=bnb_config,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation=model_config.attn_implementation,
                                                 use_cache=False,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map={"": 0})

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    # Add padding token (https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#-special-tokens-)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        r=custom_lora_config.lora_r,
        lora_alpha=custom_lora_config.lora_alpha,
        lora_dropout=custom_lora_config.lora_dropout,
        target_modules="all-linear",
        bias=custom_lora_config.lora_bias,
        task_type=custom_lora_config.lora_task_type,
    )

    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        max_seq_length=sft_config.sft_max_seq_length,
        tokenizer=tokenizer,
        packing=sft_config.sft_packing,
        formatting_func=format_instruct,
        args=train_args,
    )

    # Train
    print("Start Training")
    # trainer.train(resume_from_checkpoint="./models/evaluate/placeholder_2/checkpoint-111")
    trainer.train()
    print("End Training")

    # Perform evaluation
    print("Evaluating the final model...")
    eval_results = trainer.evaluate()

    # Print evaluation results
    print("Evaluation Results:", eval_results)

    # Save model
    trainer.save_model()
    print("Model saved")

    sys.exit(0)

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
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    merged_model.save_pretrained(f"{train_args.output_dir}/merged_model", safe_serialization=True)
    tokenizer.save_pretrained(f"{train_args.output_dir}/merged_model")


if __name__ == "__main__":
    main()
