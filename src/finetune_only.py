import torch
import json
import logging
import difflib
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.utilities import PythonREPL
from mdextractor import extract_md_blocks

seed_list = [42, 40, 38, 36, 34]
max_loop = 5
init_temperature = 0.7
model_name = "caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-adapter"
tokenizer_name = "caizefeng18/llama-3.2-3b-instruct-python-alpaca-finetune-QLoRA-adapter"
prompts_path = "../data/system_prompts.json"
questions_path = "../data/questions.json"
question_key = "9"
log_output = f"../results/outputs_finetune_only_{question_key}.log"
agent_temperatures = [0.01, 0.4, 0.8, 1.2, 1.6]
time_out = 60
python = PythonREPL()


# set up logger
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler(log_output)])


def log_llm_code(code: str):
    header = "\n----------LLM Generated Python Codes----------\n"
    tail = "\n----------End of LLM Generated Python Codes----------\n"
    logging.info(f"{header}\n{code}\n{tail}")


def log_run_result(result: str):
    header = "\n----------Execution Result----------\n"
    tail = "----------End of Execution Result----------"
    logging.info(f"{header}\n{result}\n{tail}")


def count_character_changes(code1, code2):
    # Create a SequenceMatcher instance
    matcher = difflib.SequenceMatcher(None, code1, code2)
    
    # Get the opcodes (actions like insert, delete, replace)
    changes = matcher.get_opcodes()
    
    # Count the total number of changes
    characters_changed = 0
    for tag, i1, i2, j1, j2 in changes:
        if tag in ("replace", "delete", "insert"):
            characters_changed += max(i2 - i1, j2 - j1)
    
    return characters_changed


def generate_execute_code(messages, chat, id, code_block):
    if id == 'init':
        logging.info("Generating initial codes and executing...")
    elif id == 'empty':
        logging.info("Fixing empty output...")
    else:
        logging.info(f"Agent No.{id} working on codes...")
    
    ai_msg = chat.invoke(messages)
    python_code = extract_md_blocks(ai_msg.content)[code_block]
    log_llm_code(python_code)

    a = python.run(python_code, timeout=time_out)
    log_run_result(a)
    
    return python_code, a


def set_randomness(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    

# set random seed
set_randomness(seed_list[0])

# load json files
with open(prompts_path, "r") as file:
    sys_prompts = json.load(file)
with open(questions_path, "r") as file:
    questions = json.load(file)
question = questions[question_key]


# import models and data
logging.info("Loading model...")

# set the same bnb quantization config as training
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_use_double_quant=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    low_cpu_mem_usage=True,
    use_cache=True,
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# set the same padding token as training, no actual effect
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|finetune_right_pad_id|>"
    tokenizer.padding_side = "right"

# create huggingface pipeline for langchain
logging.info("Creating initial pipeline...")
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                temperature=init_temperature)
hf = HuggingFacePipeline(pipeline=pipe)
chat = ChatHuggingFace(llm=hf, verbose=False)

# prompt
logging.info(f"Question: {question}")
init_messages = [SystemMessage(content=sys_prompts["init_system"]),
                 HumanMessage(content=question)]

# generate code
init_code, init_result = generate_execute_code(init_messages, chat, 'init', 0)


for round in range(1, len(seed_list)):
    # empty output
    if len(init_result) == 0:
        for _ in range(max_loop):
            logging.warning("LLM generated codes with empty outputs. Feeding back...")
            empty_messages = [SystemMessage(content=sys_prompts["print_system"]),
                              HumanMessage(content=f"Question: {question}\n\nYour previous codes:```python\n{init_code}\n```\n\nYou did not execute the function you created and print out the execution results. Please modify your previous outputs to print out the results.")]

            init_code, init_result = generate_execute_code(empty_messages, chat, 'empty', -1)
            
            if len(init_result) != 0:
                break

    # if empty output not corrected
    if len(init_result) == 0:
        logging.warning("Empty outputs correction failed. Changing the random seed...") # back to the beginning of the big loop
        set_randomness(seed_list[round])
        
        if round == len(seed_list)-1:
            logging.info("Debugging unsuccessful, exiting...")
            exit()
        else:
            logging.info(f"New round begins...")
            init_code, init_result = generate_execute_code(init_messages, chat, 'init', 0)

    # if output not empty
    else:
        logging.info("Output generated. Exiting...")
        exit()