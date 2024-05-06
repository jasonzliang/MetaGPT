"""
Filename: MetaGPT/examples/agent_creator.py
Created Date: Tuesday, September 12th 2023, 3:28:37 pm
Author: garylin2099
"""
import asyncio
import re

from metagpt.actions import Action
from metagpt.config2 import config
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message

EXAMPLE_CODE_FILE = METAGPT_ROOT / "examples/build_customized_agent.py"
MULTI_ACTION_AGENT_CODE_EXAMPLE = EXAMPLE_CODE_FILE.read_text()


class CreateAgent(Action):
    PROMPT_TEMPLATE: str = """
    ### BACKGROUND
    You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The PROMPT_TEMPLATE that you use for writing the code can be illustrated by the following example:

    ### EXAMPLE STARTS AT THIS LINE
    PROMPT_TEMPLATE: str = '''
    Write a python function that can {instruction}.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    '''
    ### EXAMPLE ENDS AT THIS LINE

    ### TASK
    Return an improved version of the example PROMPT_TEMPLATE that allows better, higher quality, and more accurate code to be written.

    ### OUTPUT
    PROMPT_TEMPLATE: str = '''
    """

    # System prompt override for wizardcoder LLM
    # PROMPT_TEMPLATE: str = """
    # You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The PROMPT_TEMPLATE that you use for writing the code can be illustrated by the following example:

    # PROMPT_TEMPLATE: str = '''
    # Write a python function that can {instruction}.
    # Return ```python your_code_here ``` with NO other texts,
    # your code:
    # '''

    # Create an improved, more useful and effective version of the example PROMPT_TEMPLATE that allows better, higher quality, and more accurate code to be written.

    # Your response must start with PROMPT_TEMPLATE: str = ''' and end with ''' with NO other texts.
    # """

    async def run(self, example: str, instruction: str):
        if instruction == "DEFAULT":
            prompt = self.PROMPT_TEMPLATE
        else:
            prompt = instruction

        # print(prompt);exit()
        rsp = await self._aask(prompt)
        # code_text = CreateAgent.parse_code(rsp)
        code_text = rsp
        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r"```python(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else ""
        config.workspace.path.mkdir(parents=True, exist_ok=True)
        new_file = config.workspace.path / "agent_created_agent.py"
        new_file.write_text(code_text)
        return code_text


class AgentCreator(Role):
    name: str = "Matrix"
    profile: str = "AgentCreator"
    agent_template: str = MULTI_ACTION_AGENT_CODE_EXAMPLE
    code_text: str = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([CreateAgent])


    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        msg = self.rc.memory.get()[-1]

        instruction = msg.content
        create_agent_action = CreateAgent()
        # System prompt override for wizardcoder LLM
        # create_agent_action.set_prefix("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
        self.code_text = await create_agent_action.run(example=self.agent_template,
            instruction=instruction)
        msg = Message(content=self.code_text, role=self.profile, cause_by=todo)

        return msg

    def get_code_text(self):
        return self.code_text


async def crossover(n=10):

    PROMPT_TEMPLATE_1 = '''
Write a python function that can {instruction}.

The function should follow Google's Python Style Guide (https://google.github.io/styleguide/pyguide.html) and be modular, easy to read and maintain. The function should return the result of the operation in a clear and understandable way.

Here is an example of how your function might look like:

```python
def my_function(parameter):
    """This is a one-line description of what this function does.

    Args:
        parameter: This is the explanation of the parameter.

    Returns:
        The return value and its explanation.

    Raises:
        Any exceptions that are raised and why they might occur.
    """
    # Your code here
```

Please replace `my_function`, `parameter`, `This is a one-line description of what this function does.`, `This is the explanation of the parameter.`, `The return value and its explanation.`, `Any exceptions that are raised and why they might occur.` with your actual function name, parameters, descriptions, return values, and exception handling respectively.
    '''

    PROMPT_TEMPLATE_2 = '''
### Task Description
Write a Python function that {instruction}. Ensure your code adheres to the following guidelines for quality and maintainability:

- **Modularity**: Break down the solution into smaller, reusable components where applicable.
- **Readability**: Use meaningful variable and function names that clearly indicate their purpose or the data they hold.
- **Efficiency**: Optimize for performance where necessary, avoiding unnecessary computations or memory usage.
- **Error Handling**: Include basic error handling to manage potential exceptions or invalid inputs.
- **Documentation**: Provide brief comments or a docstring explaining the logic behind key sections of your code or complex operations.
- **Testing**: Optionally, include a simple example or test case that demonstrates how to call your function and what output to expect.

### Your Code
Return your solution in the following format:
```python
# Your code here
```
with no additional text outside the code block.

### Example
If the task is to "calculate the factorial of a given number," your submission should look like this:

```python
def factorial(n):
    """Calculate the factorial of a given number n."""
    if n < 0:
        return "Error: Negative numbers do not have factorials."
    elif n == 0:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

# Example usage:
# print(factorial(5))
```
    '''

    for i in range(n):
        agent_template = MULTI_ACTION_AGENT_CODE_EXAMPLE
        creator = AgentCreator(agent_template=agent_template)

        crossover_prompt = """
Here are two prompts for code generation:

###### PROMPT 1
{prompt_1}

###### PROMPT 2
{prompt_2}

Combine and merge these two prompts to create a more effective, useful, and powerful prompt for coder generation. Be creative and try to output interesting, original, and unique prompts. Output the combined prompt below with NO other texts:

PROMPT_TEMPLATE = '''
your_output_here
'''
        """
        crossover_prompt = crossover_prompt.format(
            prompt_1=PROMPT_TEMPLATE_1, prompt_2=PROMPT_TEMPLATE_2)

        await creator.run(crossover_prompt)
        improved_prompt = creator.get_code_text()
        with open("results/improved_crossover_prompt.txt", "a") as f:
            f.write(improved_prompt)
            f.write("\n\n")


async def mutate(n=1):
    for i in range(n):
        agent_template = MULTI_ACTION_AGENT_CODE_EXAMPLE
        creator = AgentCreator(agent_template=agent_template)


        await creator.run("DEFAULT")
        improved_prompt = creator.get_code_text()
        with open("results/improved_prompt.txt", "a") as f:
            f.write(improved_prompt)
            f.write("\n\n")


if __name__ == "__main__":
    asyncio.run(crossover())

    # import asyncio

    # async def main():
    #     agent_template = MULTI_ACTION_AGENT_CODE_EXAMPLE

    #     creator = AgentCreator(agent_template=agent_template)

    #     msg = """
    #     Write an agent called SimpleTester that will take any code snippet (str) and do the following:
    #     1. write a testing code (str) for testing the given code snippet, save the testing code as a .py file in the current working directory;
    #     2. run the testing code.
    #     You can use pytest as the testing framework.
    #     """
    #     await creator.run(msg)

    # asyncio.run(main())