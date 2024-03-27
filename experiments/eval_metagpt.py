"""
Filename: MetaGPT/examples/build_customized_multi_agents.py
Created Date: Wednesday, November 15th 2023, 7:12:39 pm
Author: garylin2099
"""
import copy
import fire
import os
import pprint
import re
import sys
import time

from metagpt.actions import Action, UserRequirement
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = """
    Write a python function that can {instruction}.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """
    name: str = "SimpleWriteCode"
    code_text: str = ""

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        self.code_text = parse_code(rsp)

        return self.code_text


class SimpleWriteCode2(SimpleWriteCode):
    PROMPT_TEMPLATE: str = """
    ### TASK
    Write a Python function that {instruction}. Your function should adhere to the following guidelines:
    - Follow Google Python Style Guide: Use descriptive names, and comment generously.
    - Ensure modularity: Break down the task into smaller, reusable components if possible.
    - Prioritize readability: Write code that is easy for others to read and understand.
    - Focus on maintainability: Use clear logic and avoid unnecessary complexity.

    ### EXAMPLE
    Provide an example call to your function and its expected output.

    ### YOUR CODE
    Return your code snippet within the triple backticks below. Include any necessary comments to explain the logic and functionality of your code. Ensure there are no texts outside the backticks other than your code.

    ```python
    # your_code_here
    ```
    """

class SimpleWriteCode3(SimpleWriteCode):
    PROMPT_TEMPLATE: str = """
    Write a Python function following Google's Python style guide that accomplishes the following task: {instruction}. Ensure the code is modular, easy to read, and maintainable. Include docstrings to describe the function's purpose, parameters, and return value. Also, consider edge cases and error handling in your implementation.

    Your code should adhere to the following principles:
    - Use descriptive names for functions and variables.
    - Follow PEP 8 style guidelines for code formatting.
    - Write modular code that could be easily extended or modified.
    - Include comprehensive docstrings following Google's style guide.
    - Implement error handling where necessary.

    Return your code enclosed in triple backticks with the 'python' specifier, like so:
    ```python
    your_code_here
    ```
    Ensure there are NO other texts outside the code block. Your code:
    """

class SimpleWriteCode4(SimpleWriteCode):
    PROMPT_TEMPLATE: str = """
    ### TASK
    Write a Python function following Google's Python style guide that accomplishes the following task: {instruction}. Your function should be modular, easy to read, and maintainable. Consider edge cases and input validation as part of your implementation.

    ### REQUIREMENTS
    - Follow [Google's Python Style Guide](https://google.github.io/styleguide/pyguide.html).
    - Ensure the code is modular and can be easily extended or modified.
    - Write clean, readable code with appropriate variable names and comments.
    - Include error handling and input validation.
    - Consider efficiency and avoid unnecessary computations.

    ### OUTPUT
    Return your code enclosed in triple backticks with the 'python' specifier, like so:
    ```python
    # your_code_here
    ```
    Ensure there are no texts outside the code block. Provide a brief comment within the code to describe the functionality and any important considerations.

    ### EXAMPLE
    If your task is to 'calculate the sum of a list of numbers', your submission should look like this:
    ```python
    def sum_of_list(numbers):
        # This function calculates the sum of a list of numbers, ensuring input is a list.
        if not isinstance(numbers, list):
            raise ValueError("Input must be a list of numbers.")
        return sum(numbers)
    ```
    """

class SimpleWriteCode5(SimpleWriteCode):
    PROMPT_TEMPLATE: str = '''
    # Objective:
    # Write a Python function that accomplishes the following task: {instruction}.
    # Your function should adhere to best practices in coding, ensuring it is modular, readable, and maintainable.
    # Please include comments where necessary to explain sections of your code or logic used.

    # Instructions:
    # 1. Clearly define the function's purpose and the problem it solves.
    # 2. Use descriptive variable names that make the code easy to understand.
    # 3. Break down the problem into smaller, manageable components if necessary.
    # 4. Include error handling to manage potential exceptions or unexpected input.
    # 5. Write a brief docstring for your function, explaining what it does, its parameters, and its return value.

    # Return Format:
    # Return your code snippet in the following format:
    # ```python
    # [your_code_here]
    # ```
    # Ensure there is no additional text outside the code block.

    # Example:
    # If the instruction is "calculate the factorial of a number," your response should be structured as follows:

    ```python
    def calculate_factorial(number):
        """
        Calculate the factorial of a given number.

        Parameters:
        number (int): The number to calculate the factorial of.

        Returns:
        int: The factorial of the number.
        """
        if not isinstance(number, int) or number < 0:
            raise ValueError("The number must be a non-negative integer.")
        factorial = 1
        for i in range(1, number + 1):
            factorial *= i
        return factorial
    ```
    '''

class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([SimpleWriteCode5])

    def get_code_text(self):
        return self.actions[0].code_text

    def get_prompt_template(self):
        return self.actions[0].PROMPT_TEMPLATE


class SimpleWriteTest(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Write {k} unit tests using pytest for the given function, assuming you have imported it.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """

    name: str = "SimpleWriteTest"

    async def run(self, context: str, k: int = 3):
        prompt = self.PROMPT_TEMPLATE.format(context=context, k=k)

        rsp = await self._aask(prompt)

        code_text = parse_code(rsp)

        return code_text


class SimpleTester(Role):
    name: str = "Bob"
    profile: str = "SimpleTester"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteTest])
        # self._watch([SimpleWriteCode])
        self._watch([SimpleWriteCode, SimpleWriteReview])  # feel free to try this too

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo

        # context = self.get_memories(k=1)[0].content # use the most recent memory as context
        context = self.get_memories()  # use all memories as context

        code_text = await todo.run(context, k=5)  # specify arguments
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


class SimpleWriteReview(Action):
    PROMPT_TEMPLATE: str = """
    Context: {context}
    Review the test cases and provide one critical comments:
    """

    name: str = "SimpleWriteReview"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        return rsp


class SimpleReviewer(Role):
    name: str = "Charlie"
    profile: str = "SimpleReviewer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([SimpleWriteReview])
        self._watch([SimpleWriteTest])


async def main(
    idea: str = "write a function that calculates the product of a list",
    investment: float = 3.0,
    n_round: int = 5,
    add_human: bool = False,
):
    logger.info(idea)

    team = Team()
    team.hire(
        [
            SimpleCoder(),
            SimpleTester(),
            SimpleReviewer(is_human=add_human),
        ]
    )

    team.invest(investment=investment)
    team.run_project(idea)
    await team.run(n_round=n_round)


def generate_code_prompt(example: dict) -> str:
    return example['instruction']


async def eval_humaneval(
    n_round=5,
    result_dir="humaneval_results_%s" % int(time.time())
):
    team = Team()
    coder = SimpleCoder()
    logger.info(coder.get_prompt_template())
    # exit()
    team.hire(
        [
            coder
        ]
    )
    team.invest(investment=1e308)

    problems = get_human_eval_plus()
    eval_name = "humaneval"
    results = []

    for task_id, problem in problems.items():
        sample = {"instruction": problem['prompt'],
            "input": problem['base_input']}
        prompt = generate_code_prompt(sample)
        logger.info("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

        team.run_project(prompt)
        await team.run(n_round=n_round)
        output = coder.get_code_text()
        logger.info("#### MetaGPT Output:\n%s" % output)

        task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
        os.makedirs(task_id_dir, exist_ok=True)
        result_file = os.path.join(task_id_dir, "0.py")
        with open(result_file, 'w') as f:
            f.write(output)

        results.append({'task_id': task_id, 'solution': output})

    os.system("evalplus.evaluate --dataset %s --samples %s | tee %s"
        % (eval_name, result_dir, os.path.join(result_dir, "evalplus.txt")))
    # with open(os.path.join(result_dir, "config.txt"), "w") as f:
    #     f.write(pprint.pformat(locals(), compact=True))
    with open(os.path.join(result_dir, "prompt_template.txt"), "w") as f:
        f.write(coder.get_prompt_template())

if __name__ == "__main__":
    # eval_humaneval()
    fire.Fire(eval_humaneval)
    # fire.Fire(main)
