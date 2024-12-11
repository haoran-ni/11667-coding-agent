# Self-debugging coding agent

A coding agent finetuned from `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` capable of debugging with self-iterations.

## Running
The running tests are done on NERSC. Please make sure `cudatoolkit/12.4` is loaded before setting up the conda environment.
```
conda create --name cmu-llms-final python=3.11
conda activate cmu-llms-final
pip install -r requirements.txt

login wandb
huggingface-cli login
```
Please note that the environment may be updated frequently, so be sure to check and update your conda environment when there are new commits to the repo.

### Fine-tuning
To fine-tune the model, run the following command:
```shell
python src/finetuning.py configs/<training_config_file>
```

### Evaluation
To evaluate the model, run the following command:
```shell
python src/inference.py configs/<inference_config_file> <name_or_path_to_the_model>
```

## Hugging Face Repository
This project is hosted on Hugging Face under the repository:
### Adapters (Better Performance)
- [`caizefeng18/llama-3.2-1b-instruct-python-alpaca-finetune-QLoRA-adapter`](https://huggingface.co/caizefeng18/llama-3.2-1b-instruct-python-alpaca-finetune-QLoRA-adapter)
- [`caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-adapter`](https://huggingface.co/caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-adapter)
### Merged Models
- [`caizefeng18/llama-3.2-1b-instruct-python-alpaca-finetune-QLoRA-merged`](https://huggingface.co/caizefeng18/llama-3.2-1b-instruct-python-alpaca-finetune-QLoRA-merged)
- [`caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-merged`](https://huggingface.co/caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-merged)
