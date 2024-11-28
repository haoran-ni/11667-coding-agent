import torch
import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Merge LoRA and base model for causal language modeling.")
    parser.add_argument(
        "model_name",
        type=str,
        help="Name or path of the base model."
    )
    args = parser.parse_args()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.padding_side = "right"

    # Load the saved model
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_name,
        low_cpu_mem_usage=True,  # consistent with training, otherwise slightly degraded performance
        torch_dtype=torch.bfloat16,  # consistent with training
        device_map={"": 0},
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()

    # Save the merged model
    save_path = f"{args.model_name}/merged_model"
    merged_model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)

    print(f"Merged model and tokenizer saved to {save_path}")

if __name__ == "__main__":
    main()
