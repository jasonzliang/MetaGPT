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

class SimpleWriteCodeImproved(SimpleWriteCode):
    PROMPT_TEMPLATE: str = '''
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

class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([SimpleWriteCode])

    def set_env(self, env: "Environment"):
        """Set the environment in which the role works. The role can talk to the environment and can also receive
        messages by observing."""
        self.rc.env = env
        if env:
            env.set_addresses(self, self.addresses)
            self.llm.system_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            self.llm.cost_manager = self.context.cost_manager
            self.set_actions(self.actions)  # reset actions to update llm and prefix

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


def create_new_team():
    team = Team()
    coder = SimpleCoder()
    logger.info(coder.get_prompt_template())
    team.hire(
        [
            coder
        ]
    )
    team.invest(investment=1e308)
    return team, coder


async def eval_humaneval(
    n_round=5,
    result_dir="humaneval_results_%s" % int(time.time())
):


    problems = get_human_eval_plus()
    eval_name = "humaneval"
    results = []

    for task_id, problem in problems.items():
        sample = {"instruction": problem['prompt'],
            "input": problem['base_input']}
        prompt = generate_code_prompt(sample)
        logger.info("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

        team, coder = create_new_team()
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