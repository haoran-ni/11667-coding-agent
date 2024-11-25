# def format_user_text(instruction, input):
#     return f"""### Instruction:
# Use the Task below and the Input given to write the Response, which is a Python programming code that can solve the following Task:
#
# ### Task:
# {instruction}
#
# ### Input:
# {input}
#
# ### Response:
# """
#
# def format_assistant_text(output):
#     return f"""```python
# {output}
# ```
# """

def format_user_text(instruction, input):
    return f"""### Instruction:
Use the Task below and the Input given to write the Response, which is a Python programming code that can solve the following Task:

### Task:
{instruction}

### Input:
{input if input != '' else 'Not applicable'}
"""

def format_assistant_text(output):
    return f"""### Response:
```python
{output}
```
"""

# def format_user_text(instruction, input):
#     return f"""### INSTRUCTION:
# Using the provided TASK and INPUT, generate a Python program as the RESPONSE that effectively solves the given TASK based on the specified INPUT.
#
# ### TASK:
# {instruction}
#
# ### INPUT:
# {input}
# """
#
# def format_assistant_text(output):
#     return f"""### RESPONSE:
# ```python
# {output}
# ```
# """