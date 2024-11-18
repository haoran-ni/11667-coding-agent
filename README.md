# Self-debugging coding agent

A coding agent finetuned from <base_model_name_to_add>, capable of debugging with self-iterations.

## Running
The running tests are done on NERSC. Please make sure `cudatoolkit/12.4` is loaded before setting up conda environment.
```
conda create --name cmu-llms-final python=3.11
conda activate cmu-llms-final
pip install -r requirements.py
login wandb
```
Please note that the environment may be updated frequently, so be sure to check and update your conda environment when there are new commits to the repo.

