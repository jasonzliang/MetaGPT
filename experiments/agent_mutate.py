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
    # PROMPT_TEMPLATE: str = """
    # ### BACKGROUND
    # You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The PROMPT_TEMPLATE that you use for writing the code can be illustrated by the following example:

    # ### EXAMPLE STARTS AT THIS LINE
    # PROMPT_TEMPLATE: str = '''
    # Write a python function that can {instruction}.
    # Return ```python your_code_here ``` with NO other texts,
    # your code:
    # '''
    # ### EXAMPLE ENDS AT THIS LINE

    # ### TASK
    # Return an improved version of the example PROMPT_TEMPLATE that allows better, higher quality, and more accurate code to be written.

    # ### OUTPUT
    # PROMPT_TEMPLATE: str = '''
    # """

    # System prompt override for wizardcoder LLM
    PROMPT_TEMPLATE: str = """
    You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The PROMPT_TEMPLATE that you use for writing the code can be illustrated by the following example:

    PROMPT_TEMPLATE: str = '''
    Write a python function that can {instruction}.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    '''

    Return an improved version of the example PROMPT_TEMPLATE that allows better, higher quality, and more accurate code to be written. Your response must start with PROMPT_TEMPLATE: str = ''' and end with '''.
    """

    async def run(self, example: str, instruction: str):
        # prompt = self.PROMPT_TEMPLATE.format(example=example, instruction=instruction)
        prompt = self.PROMPT_TEMPLATE

        rsp = await self._aask(prompt)

        code_text = CreateAgent.parse_code(rsp)

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
        create_agent_action.set_prefix("Below is an instruction that describes a task. Write a response that appropriately completes the request.")
        self.code_text = await create_agent_action.run(example=self.agent_template,
            instruction=instruction)
        msg = Message(content=self.code_text, role=self.profile, cause_by=todo)

        return msg

    def get_code_text(self):
        return self.code_text


async def mutate(n=1):
    for i in range(n):
        agent_template = MULTI_ACTION_AGENT_CODE_EXAMPLE
        creator = AgentCreator(agent_template=agent_template)

        msg = \
"""
Role: You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. Here is the PROMPT_TEMPLATE you use for writing the code.

PROMPT_TEMPLATE:
Write a python function that can {instruction}.
Return ```python your_code_here ``` with NO other texts,
your code:

Return an improved version of the current PROMPT_TEMPLATE that will allow for better, higher quality, and more accurate code to be written.

PROMPT_TEMPLATE:
"""
        await creator.run(msg)
        improved_prompt = creator.get_code_text()
        with open("improved_prompt.txt", "a") as f:
            f.write(improved_prompt)
            f.write("\n\n")


if __name__ == "__main__":
    asyncio.run(mutate())

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