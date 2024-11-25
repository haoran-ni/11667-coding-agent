import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_experimental.utilities import PythonREPL
from mdextractor import extract_md_blocks


def print_llm_code(code: str):
    header = "----------LLM Generated Python Codes----------"
    tail = "----------End of LLM Generated Python Codes----------"
    print(f"{header}\n{code}\n{tail}")


def print_run_result(result: str):
    header = "----------Running Result----------"
    tail = "----------End of Running Result----------"
    print(f"{header}\n{result}\n{tail}")


device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer_name = "meta-llama/Llama-3.2-1B-Instruct"
with open("system_prompts.json", "r") as file:
    sys_prompts = json.load(file)

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# create huggingface pipeline for langchain
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device=device)
hf = HuggingFacePipeline(pipeline=pipe)
chat = ChatHuggingFace(llm=hf, verbose=False)

# prompt
messages = [
    SystemMessage(content=sys_prompts["init_system"]),
    HumanMessage(content="We have a function f(x)=x^2 + 3*x + 4, find the values of f(3), f(4), and f(5).")
]

# generate code
ai_msg = chat.invoke(messages)
python_code = extract_md_blocks(ai_msg.content)[0]
print_llm_code(python_code)

# run code
python = PythonREPL()
a = python.run(python_code, timeout=30)
print_run_result(a)


# If beyond certain steps and the bug is still not fixed
# instead of keep fixing the bug, the agent will just start over, do not keep working on codes with no promising ends