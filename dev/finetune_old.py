import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, HfArgumentParser, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class ModelConfig:
    model_name: str
    dataset_name: str


def main():
    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config
    
    
    # parse configuration arguments
    hfparser = HfArgumentParser((ModelConfig, TrainingArguments))
    model_config, train_args = hfparser.parse_json_file(json_file=config_path)
    # lora configurations (there are bugs using HfArgumentParser, so it is explicitly configured here)
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             inference_mode=False,
                             r=8,
                             lora_alpha=32,
                             lora_dropout=0.1)
    
    # retrieve dataset
    train_dataset = load_dataset(model_config.dataset_name, split= )
    eval_dataset = load_dataset(model_config.dataset_name, split= )
    
    # retrieve model
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    
    # add padding token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    # add data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # THIS TOKENIZATION PART HAS BUGS. I TRIED ON A TOY DATASET AND KEPT RECEIVING KEYWORD ERRORS
    # tokenization
    def tokenize_func(examples):
        return tokenizer(
            examples['[FEATURE_NAME_TODO]'],
            truncation=True,
            max_length=512
        )
    tokenized_train = train_dataset.map(tokenize_func, batched=True, remove_columns=['[FEATURE_NAME_TODO]'])
    tokenized_eval = eval_dataset.map(tokenize_func, batched=True, remove_columns=['[FEATURE_NAME_TODO]'])
    
    
    # peft
    model = get_peft_model(model, peft_config)
    # print out trainable parameters after applying LoRA
    model.print_trainable_parameters()
    
    # training
    trainer = Trainer(
        model = model,
        args = train_args,
        train_dataset = tokenized_train,
        eval_dataset = tokenized_eval,
        data_collator = data_collator,
        tokenizer = tokenizer
    )
    
    trainer.train()
    trainer.save_model(train_args.output_dir)
    
    # evaluation
    # TODO
    


if __name__ == "__main__":
    main()