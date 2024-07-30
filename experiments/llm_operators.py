import asyncio
import copy
import glob
import json
import logging
import os
import platform
import pprint
import re
import sys
import random
import traceback
import time

from metagpt.actions import Action, UserRequirement
from metagpt.config2 import Config
from metagpt.logs import logger as mlogger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

from pathos.pools import ProcessPool
from retry import retry

from alg_util import randomword
from alg_util import MIN_FITNESS, EPSILON, ID_LENGTH, MIN_POP_SIZE
from autogen_builder import autogen_mutate, autogen_crossover
from autogen_builder import BUILDER_LLM_CONFIG
from util import extract_evalplus, parse_code, parse_prompt_template

DEFAULT_MAIN_ROLE = \
"""Write a python function that can {instruction}.
Return ```python your_code_here ``` with NO other texts,
your code:
"""


class MutateAction(Action):
    PROMPT_TEMPLATE: str = \
"""You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The prompt template that you use for writing the code can be illustrated by the following example:

{prompt}

Return an improved version of the example prompt template which writes better code that is more efficient, accurate, and correct. Output the improved prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

PROMPT_TEMPLATE: str = '''
your_output_here
'''
"""
    PROMPT_TEMPLATE_2: str = \
"""You are a professional engineer; the prompt template that you use for writing the code can be illustrated by the following example:

{prompt}

In addition, here are some incorrect code that the prompt template has generated:

### INCORRECT CODE START HERE
{negative_examples}
### INCORRECT CODE END HERE

Return an improved version of the example prompt template which writes efficient, accurate, and correct code. The prompt must be able to avoid writing the incorrect code above. Output the improved prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

PROMPT_TEMPLATE: str = '''
your_output_here
'''
"""
    name: str = "MutateAction"
    code_text: str = ""

    async def run(self, prompt: str):
        prompt = self.PROMPT_TEMPLATE.format(prompt=prompt)
        code_text = await self._aask(prompt)
        self.code_text = parse_prompt_template(code_text)
        return self.code_text

    async def run2(self, prompt: str, negative_examples: str):
        prompt = self.PROMPT_TEMPLATE_2.format(prompt=prompt,
            negative_examples=negative_examples)
        code_text = await self._aask(prompt)
        self.code_text = parse_prompt_template(code_text)
        return self.code_text

    def get_code_text(self):
        return self.code_text


class CrossoverAction(Action):
    PROMPT_TEMPLATE: str = \
"""You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. Here are two example prompt templates that you use for writing code:

### PROMPT TEMPLATE 1
{prompt_1}

### PROMPT TEMPLATE 2
{prompt_2}

Combine and merge these two prompt templates to create a better prompt for writing more efficient, accurate, and correct code. Try to output interesting, original, and creative prompts. Output the combined prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

PROMPT_TEMPLATE: str = '''
your_output_here
'''
"""
    PROMPT_TEMPLATE_2: str = \
"""You are a professional engineer; here is the main prompt template that you use for writing code:

{prompt}

In addition, here are additional prompt templates that are ranked in order from best to worst:

### ADDITIONAL PROMPT TEMPLATES START HERE
{additional_prompts}
### ADDITIONAL PROMPT TEMPLATES END HERE

Combine and merge elements from these additional prompt templates into the main prompt template that you use for writing code. Make sure to account of the ranking of each additional prompt. Try to create interesting, original, and creative prompts that can write efficient, accurate, and correct code. Output the combined prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

PROMPT_TEMPLATE: str = '''
your_output_here
'''
"""
    name: str = "CrossoverAction"
    code_text: str = ""

    async def run(self, prompt_1: str, prompt_2: str):
        prompt = self.PROMPT_TEMPLATE.format(
            prompt_1=prompt_1, prompt_2=prompt_2)
        code_text = await self._aask(prompt)
        self.code_text = parse_prompt_template(code_text)
        return self.code_text

    async def run2(self, prompt: str, additional_prompts: str):
        prompt = self.PROMPT_TEMPLATE_2.format(
            prompt=prompt, additional_prompts=additional_prompts)
        code_text = await self._aask(prompt)
        self.code_text = parse_prompt_template(code_text)
        return self.code_text

    def get_code_text(self):
        return self.code_text


class SimpleWriteCode(Action):
    PROMPT_TEMPLATE: str = ""
    name: str = "SimpleWriteCode"
    code_text: str = ""

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE
        try:
            prompt = prompt.format(instruction=instruction)
        except:
            # If {instruction} not found, search for first pair of braces
            special_word = prompt[prompt.find("{"):prompt.find("}")+1]
            prompt = prompt.replace(special_word, instruction)
        # finally:
        rsp = await self._aask(prompt)
        self.code_text = parse_code(rsp)
        return self.code_text


class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "prompt optimization engineer"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.set_actions([SimpleWriteCode])

    # System prompt override for wizardcoder LLM
    # def _get_prefix(self):
    #     return "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    def get_code_text(self):
        return self.actions[0].code_text

    def get_prompt_template(self):
        return self.actions[0].PROMPT_TEMPLATE

    def set_prompt_template(self, new_prompt):
        self.actions[0].PROMPT_TEMPLATE = new_prompt


def create_new_team(llm_config):
    try: # Hack to get it running on M1 mac
        loop = asyncio.get_event_loop()
    except RuntimeError as e:
        if str(e).startswith('There is no current event loop in thread'):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            raise

    config = Config.default()
    config.llm.model = llm_config.get("model", "gpt-3.5-turbo")
    config.llm.temperature = llm_config.get("temperature", 0.2)
    config.llm.top_p = llm_config.get("top_p", 1.0)

    team = Team()
    coder = SimpleCoder(config=config)
    team.hire([coder])
    team.invest(investment=1e308)
    return team, coder


### METAGPT SINGLE ROLE MUTATION/CROSSOVER ###
@retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
    logger=logging.getLogger('evolve_role'))
def llm_mutate(prompt, llm_config):
    config = Config.default()
    config.llm.model = llm_config.get("model", "gpt-4-turbo")
    config.llm.temperature = llm_config.get("temperature", 1.0)
    config.llm.top_p = llm_config.get("top_p", 1.0)

    mutate_operator = MutateAction(config=config)
    improved_prompt = asyncio.run(mutate_operator.run(prompt=prompt))

    return improved_prompt


@retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
    logger=logging.getLogger('evolve_role'))
def llm_mutate2(prompt, result_dir, n=3, llm_config={}):
    config = Config.default()
    config.llm.model = llm_config.get("model", "gpt-4-turbo")
    config.llm.temperature = llm_config.get("temperature", 1.0)
    config.llm.top_p = llm_config.get("top_p", 1.0)

    eval_json_fp = os.path.join(result_dir, "eval_results.json")
    assert os.path.exists(eval_json_fp)
    with open(eval_json_fp, "r") as f:
        eval_json = json.load(f)

    negative_examples = []
    for key, solution_dict in eval_json['eval'].items():
        solution_dict = solution_dict[0]
        if solution_dict["base_status"] == "fail":
            negative_examples.append(solution_dict["solution"])
    if len(negative_examples) == 0:
        negative_examples = ["No negative examples exist!"]
    elif len(negative_examples) > n:
        negative_examples = random.sample(negative_examples, n)
    negative_examples = "\n".join(negative_examples)

    mutate_operator = MutateAction(config=config)
    improved_prompt = asyncio.run(mutate_operator.run2(prompt=prompt,
        negative_examples=negative_examples))

    return improved_prompt


@retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
    logger=logging.getLogger('evolve_role'))
def llm_crossover(prompt_1, prompt_2, llm_config):
    config = Config.default()
    config.llm.model = llm_config.get("model", "gpt-4-turbo")
    config.llm.temperature = llm_config.get("temperature", 1.0)
    config.llm.top_p = llm_config.get("top_p", 1.0)

    crossover_operator = CrossoverAction(config=config)
    improved_prompt = asyncio.run(
        crossover_operator.run(prompt_1=prompt_1, prompt_2=prompt_2))

    return improved_prompt


@retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
    logger=logging.getLogger('evolve_role'))
def llm_crossover2(prompt, additional_prompts, llm_config):
    config = Config.default()
    config.llm.model = llm_config.get("model", "gpt-4-turbo")
    config.llm.temperature = llm_config.get("temperature", 1.0)
    config.llm.top_p = llm_config.get("top_p", 1.0)

    if len(additional_prompts) == 0:
        additional_prompts = ["No additional prompts exist!"]
    additional_prompts = "\n".join(additional_prompts)

    crossover_operator = CrossoverAction(config=config)
    improved_prompt = asyncio.run(crossover_operator.run2(prompt=prompt,
        additional_prompts=additional_prompts))

    return improved_prompt


### AUTOGEN TEAM MUTATION/CROSSOVER ###
# @retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
#     logger=logging.getLogger('evolve_role'))
def llm_mutate_team(team_role, llm_config):
    assert type(team_role) is dict
    builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
    builder_llm_config.update(llm_config.get("builder_llm_config", {}))

    if 'building_task' in team_role: del team_role['building_task']
    agent_list, agent_configs, builder, builder_dict = autogen_mutate(
        builder_cfg=team_role,
        output_cfg=None,
        builder_llm_config=builder_llm_config,
        eval_mode=True,
        work_dir="/tmp/%s" % randomword(ID_LENGTH))
    builder.clear_all_agents(recycle_endpoint=False)
    if 'building_task' in builder_dict: del builder_dict['building_task']
    return builder_dict


# @retry(Exception, tries=-1, delay=1, max_delay=16, backoff=2,
#     logger=logging.getLogger('evolve_role'))
def llm_crossover_team(team_role, other_team_role, llm_config):
    assert type(team_role) is dict; assert type(other_team_role) is dict
    builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
    builder_llm_config.update(llm_config.get("builder_llm_config", {}))

    if 'building_task' in team_role: del team_role['building_task']
    agent_list, agent_configs, builder, builder_dict = autogen_crossover(
        builder_cfgs=[team_role, other_team_role],
        output_cfg=None,
        builder_llm_config=builder_llm_config,
        eval_mode=True,
        work_dir="/tmp/%s" % randomword(ID_LENGTH))
    builder.clear_all_agents(recycle_endpoint=False)
    if 'building_task' in builder_dict: del builder_dict['building_task']
    return builder_dict


#### Unit tests ####
PROMPT_TEMPLATE_1 = DEFAULT_MAIN_ROLE
PROMPT_TEMPLATE_2 = """### Task Description
Write a Python function that {instruction}. Ensure your code adheres to the following guidelines for quality and maintainability:

- **Modularity**: Break down the solution into smaller, reusable components where applicable.
- **Readability**: Use meaningful variable and function names that clearly indicate their purpose or the data they hold.

### Your Code
Return your solution in the following format:
```python your_code_here ```
with no additional text outside the code block.
"""


def _test_mutation_crossover(test_err=False):
    llm_model = 'N/A' if test_err else 'gpt-4o'
    llm_config = {'model': llm_model, 'temperature': 1.2, 'top_p': 1.0}

    try:
        output = llm_mutate(PROMPT_TEMPLATE_1, llm_config=llm_config)
        print("### LLM_MUTATE RETURN VALUE ###")
        print(output)
        print("###############################")
    except:
        traceback.print_exc()

    try:
        output = llm_crossover(PROMPT_TEMPLATE_1, PROMPT_TEMPLATE_2,
            llm_config=llm_config)
        print("### LLM_CROSSOVER RETURN VALUE ###")
        print(output)
        print("##################################")
    except:
        traceback.print_exc()


def _test_mutation_crossover2(test_err=False):
    result_dirs = sorted(glob.glob('results/**/humaneval_*'))
    assert len(result_dirs) > 0; result_dir = result_dirs[0]; print(result_dir)
    llm_model = 'N/A' if test_err else 'gpt-4o'
    llm_config = {'model': llm_model, 'temperature': 1.2, 'top_p': 1.0}

    try:
        output = llm_mutate2(PROMPT_TEMPLATE_1, result_dir,
            llm_config=llm_config, n=3)
        print("### LLM_MUTATE RETURN VALUE ###")
        print(output)
        print("###############################")
    except:
        traceback.print_exc()

    try:
        output = llm_crossover2(PROMPT_TEMPLATE_1, [PROMPT_TEMPLATE_2] * 5,
            llm_config=llm_config)
        print("### LLM_CROSSOVER RETURN VALUE ###")
        print(output)
        print("##################################")
    except:
        traceback.print_exc()


def _test_autogen_mutation_crossover(
    team_role="autogen_builder_cfg.json",
    other_team_role='autogen_mutate.json'):
    # llm_model = 'N/A' if test_err else 'gpt-4o'
    # llm_config = {'model': llm_model, 'temperature': 1.2, 'top_p': 1.0}
    llm_config = {'temperature': 1.0,
        'builder_model': 'gpt-4o', 'agent_model': 'gpt-4o', "cache_seed": None}
    with open(team_role, 'r') as f: team_role = json.load(f)
    with open(other_team_role, 'r') as f: other_team_role = json.load(f)

    try:
        output = llm_mutate_team(team_role, llm_config=llm_config)
        print("### LLM_MUTATE RETURN VALUE ###")
        pprint.pprint(output)
        print("###############################")
    except:
        traceback.print_exc()

    try:
        output = llm_crossover_team(team_role, other_team_role,
            llm_config=llm_config)
        print("### LLM_CROSSOVER RETURN VALUE ###")
        pprint.pprint(output)
        print("##################################")
    except:
        traceback.print_exc()


def _test_prompt_extractor():
    PROMPT_TEMPLATE = """PROMPT_TEMPLATE: str = '''%s'''""" % PROMPT_TEMPLATE_1
    print("Original prompt template:")
    print(PROMPT_TEMPLATE)
    print("Extracted prompt template:")
    print(parse_prompt_template(PROMPT_TEMPLATE))


if __name__ == "__main__":
    _test_autogen_mutation_crossover()
