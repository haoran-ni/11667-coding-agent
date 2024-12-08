def format_user_text(instruction, input):
    if input and input != 'Not applicable':
        return instruction
    else:
        return f"""{instruction}
{input}"""


def format_assistant_text(output):
    return f"""```python
{output}
```
"""


def format_user_text_1b(instruction, input):
    prompt = f"""You are a Python code generation assistant. You will be given an instruction and possibly an additional input. Your job is to write Python code that accomplishes the instruction, using the input if provided. Do not write anything except the Python code. Do not include explanations, reasoning, or commentary. Your entire answer must be Python code inside a single code block formatted as:
    
```python
# Your code here
```

Only produce the Python code that solves the given instruction. Do not include any additional text outside of the code block.
"""

    if input and input != 'Not applicable':
        return f"""{prompt}

### Instruction:
{instruction}"""
    else:
        return f"""{prompt}

### Instruction:
{instruction}

### Input:
{input}"""


def format_user_text_3b(instruction, input):
    prompt = f"""You are a Python code generator. Given an instruction and possibly an input, produce Python code that implements the instruction using the input if provided. Your output must be exclusively Python code, wrapped in a triple backtick Python code block with no additional text or commentary.
    
Format strictly as:

```python
# code
```"""

    if input and input != 'Not applicable':
        return f"""{prompt}

### Instruction:
{instruction}"""
    else:
        return f"""{prompt}

### Instruction:
{instruction}

### Input:
{input}"""
