import argparse
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, \
    BitsAndBytesConfig, pipeline, LlamaForCausalLM

from configuration import ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, EvaluationConfig
from templates import format_user_text, format_assistant_text, format_user_text_1b, format_user_text_3b


def main():
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")  # Positional argument
    parser.add_argument("model_name", type=str, help="Path to the model")
    parser.add_argument("output_file", type=str, help="Path to the output destination file")
    args = parser.parse_args()
    config_path = args.config
    os.environ["WANDB_PROJECT"] = "11667-llms-hw6-sft"

    # Parse configuration arguments
    hfparser = HfArgumentParser(
        (ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments, EvaluationConfig))
    model_config, custom_bnb_config, custom_lora_config, sft_config, train_args, eval_config = hfparser.parse_json_file(
        json_file=config_path)  # type: ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, TrainingArguments, EvaluationConfig

    # Retrieve dataset
    full_dataset = load_dataset(model_config.dataset_name, split=model_config.dataset_split)

    # Split dataset into train (95%), validation (4%), and test (1%)
    train_val_split = full_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)

    val_test_split = train_val_split['test'].train_test_split(test_size=0.2, shuffle=True, seed=42)

    train_dataset = train_val_split['train']
    eval_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    if not eval_config.eval_on_validation:
        eval_dataset = test_dataset

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=custom_bnb_config.load_in_4bit,
        bnb_4bit_quant_type=custom_bnb_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=custom_bnb_config.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=custom_bnb_config.bnb_4bit_use_double_quant
    )

    # Retrieve model
    if eval_config.is_unmerged and eval_config.is_bnb:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            use_cache=True,  # impact speed only, use it when doing inference
            attn_implementation=model_config.attn_implementation,  # consistent with training
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )  # type: LlamaForCausalLM

    elif eval_config.is_unmerged and not eval_config.is_bnb:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

    elif not eval_config.is_unmerged and eval_config.is_bnb:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation=model_config.attn_implementation,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "right"

    # format_user_text_func = format_user_text
    # format_user_text_func = format_user_text_1b
    format_user_text_func = format_user_text_3b

    # Format the dataset to be used in the instruct model
    def format_message(dataset):
        message_json = []
        target_response_json = []
        for i in range(len(dataset['instruction'])):
            user_text = format_user_text_func(dataset['instruction'][i], dataset['input'][i])
            assistant_text = format_assistant_text(dataset['output'][i])
            user_text_json = [{"role": "user", "content": user_text}]
            message_json.append(user_text_json)
            target_response_json.append(assistant_text)
        return message_json, target_response_json

    eval_select = eval_dataset.shuffle(seed=42)

    messages_select, target_responses = format_message(eval_select)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    data_to_save = []
    for i, message in enumerate(tqdm(messages_select)):
        # Use `Pipeline`
        outputs = pipe(
            message,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        data_to_save.append({
            "generated": outputs[0]["generated_text"][-1]["content"],
            "target": target_responses[i]
        })

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)


if __name__ == "__main__":
    main()
