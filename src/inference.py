import argparse
import os
import sys

import math
import torch
from datasets import load_dataset
from peft import LoraConfig, AutoPeftModelForCausalLM, PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, HfArgumentParser, \
    BitsAndBytesConfig, pipeline, LlamaForCausalLM
from trl import SFTTrainer

from configuration import ModelConfig, CustomBnBConfig, CustomLoRAConfig, SFTConfig, EvaluationConfig
from qdq_merge import dequantize_model
from templates import format_user_text, format_assistant_text


def main():
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file")  # Positional argument
    parser.add_argument("model_name", type=str, help="Path to the model")
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
    if eval_config.is_unmerged and eval_config.is_bnb:

        # Equivalent #1:
        # model = AutoPeftModelForCausalLM.from_pretrained(
        #     args.model_name,
        #     quantization_config=bnb_config,
        #     low_cpu_mem_usage=True,
        #     use_cache=True,  # impact speed only, use it when doing inference
        #     attn_implementation=model_config.attn_implementation,  # consistent with training
        #     torch_dtype=torch.bfloat16,
        #     device_map={"": 0},
        # )  # type: PeftModel

        # Equivalent #2:
        # peft_config = PeftConfig.from_pretrained(args.model_name)
        # base_model_path = peft_config.base_model_name_or_path
        #
        # print(f"Base model path: {base_model_path}")
        #
        # model = AutoModelForCausalLM.from_pretrained(
        #     base_model_path,
        #     quantization_config=bnb_config,
        #     low_cpu_mem_usage=True,
        #     use_cache=True,  # impact speed only, use it when doing inference
        #     attn_implementation=model_config.attn_implementation,  # consistent with training
        #     torch_dtype=torch.bfloat16,
        #     device_map={"": 0},
        # )
        #
        # model = PeftModel.from_pretrained(model=dequantize_model(model), model_id=args.model_name)

        # Equivalent #3:
        # model = AutoModelForCausalLM.from_pretrained(
        #     "base_model",
        #     quantization_config=bnb_config,
        #     attn_implementation=model_config.attn_implementation,
        #     use_cache=True,
        #     torch_dtype=torch.bfloat16,
        #     device_map={"": 0})
        # model = PeftModel.from_pretrained(model, args.model_name)  # type: PeftModel

        # Equivalent #4 (modern) (except that the model is not of type PeftModel):
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

    # Dummy Lora config to make the Trainer instantiation work for fully-quantized models
    if not eval_config.is_unmerged and eval_config.is_bnb:
        peft_config = LoraConfig(
            r=custom_lora_config.lora_r,
            lora_alpha=custom_lora_config.lora_alpha,
            lora_dropout=custom_lora_config.lora_dropout,
            bias=custom_lora_config.lora_bias,
            task_type=custom_lora_config.lora_task_type,
        )

        # Create the trainer for evaluation
        trainer = SFTTrainer(
            model=model,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            max_seq_length=sft_config.sft_max_seq_length,
            tokenizer=tokenizer,
            packing=sft_config.sft_packing,
            formatting_func=format_instruct,
            args=train_args,
        )

    else:
        # Create the trainer for evaluation
        trainer = SFTTrainer(
            model=model,
            eval_dataset=eval_dataset,
            max_seq_length=sft_config.sft_max_seq_length,
            tokenizer=tokenizer,
            packing=sft_config.sft_packing,
            formatting_func=format_instruct,
            args=train_args,
        )

    # Perform evaluation
    print("Evaluating the final model...")
    eval_results = trainer.evaluate()

    # Print evaluation results
    print("Evaluation Results:", eval_results)
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.3f}")

    if not eval_config.ppl_only:

        # Reload the model if it's a merged fully quantized model
        # To remove the dummy peft adapter added when instantiating the Trainer
        if not eval_config.is_unmerged and eval_config.is_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                use_cache=True,
                attn_implementation=model_config.attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map={"": 0}
            )

        # Format the dataset to be used in the instruct model
        def format_message(dataset):
            message_json = []
            target_response_json = []
            for i in range(len(dataset['instruction'])):
                user_text = format_user_text(dataset['instruction'][i], dataset['input'][i])
                assistant_text = format_assistant_text(dataset['output'][i])
                user_text_json = [{"role": "user", "content": user_text}]
                message_json.append(user_text_json)
                target_response_json.append(assistant_text)
            return message_json, target_response_json

        eval_select = eval_dataset.shuffle(seed=42).select(range(10))

        messages_select, target_responses = format_message(eval_select)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        model.generation_config.pad_token_id = tokenizer.pad_token_id

        if not eval_config.legacy_generation:
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

        for i, message in enumerate(messages_select):

            if eval_config.legacy_generation:
                # Use `AutoModelForCausalLM`
                input_ids = tokenizer.apply_chat_template(
                    message,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    tokenize=True
                ).to(model.device)

                outputs = model.generate(
                    input_ids,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            else:
                # Use `Pipeline`
                outputs = pipe(
                    message,
                    max_new_tokens=2048,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )

            print("--------------------------------------------------")
            print("--------------------------------------------------")
            print(message[0]['content'])
            print("--------------------------------------------------")
            print("Generated Response:")
            if eval_config.legacy_generation:
                response = outputs[0][input_ids.shape[-1]:]
                print(tokenizer.decode(response, skip_special_tokens=True))
            else:
                print(outputs[0]["generated_text"][-1]["content"])
            print("--------------------------------------------------")
            print("Target Response:")
            print(target_responses[i])


if __name__ == "__main__":
    main()
