# def format_user_text(instruction, input):
#     return f"""### Instruction:
# Use the Task below and the Input given to write the Response, which is a Python programming code that can solve the following Task:
#
# ### Task:
# {instruction}
#
# ### Input:
# {input if input != '' else 'Not applicable'}
# """
#
# def format_assistant_text(output):
#     return f"""### Response:
# ```python
# {output}
# ```
# """

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