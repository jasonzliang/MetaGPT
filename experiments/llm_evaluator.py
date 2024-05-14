import asyncio
import copy
import importlib
import logging
import os
import re
import sys
import random
import time

from metagpt.actions import Action, UserRequirement
from metagpt.config2 import Config
from metagpt.logs import logger
from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.team import Team

from pathos.pools import _ProcessPool as Pool

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl

from util import extract_evalplus_score


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


def parse_prompt_template(rsp):
    pattern = r"PROMPT_TEMPLATE: str = '''(.*)'''"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    code_text = code_text.lstrip().rstrip()
    return code_text


class MutateAction(Action):
    PROMPT_TEMPLATE: str = \
"""
You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. The prompt template that you use for writing the code can be illustrated by the following example:

{prompt}

Return an improved version of the example prompt template that allows better, higher quality, and more accurate code to be written. Output the improved prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

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

    def get_code_text(self):
        return self.code_text


class CrossoverAction(Action):
    PROMPT_TEMPLATE: str = \
"""
You are a professional engineer; the main goal is to write google-style, elegant, modular, easy to read and maintain code. Here are two prompt templates that you use for writing code:

### PROMPT TEMPLATE 1 ###
{prompt_1}

### PROMPT TEMPLATE 2 ###
{prompt_2}

Combine and merge these two prompt templates to create a more effective, useful, and powerful prompt for writing code. Be creative and try to output interesting, original, and unique prompts. Output the combined prompt template below with NO other texts and make sure the keyword "instruction" is present within the output:

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
        finally:
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


def create_new_team(llm_model):
    llm_config = Config.default()
    llm_config.llm.model = llm_model

    team = Team()
    coder = SimpleCoder(config=llm_config)
    team.hire([coder])
    team.invest(investment=1e308)
    return team, coder


def llm_mutate(prompt):
    llm_config = Config.default()
    llm_config.llm.model = "gpt-4-turbo"
    llm_config.llm.temperature = 0.7
    mutate_operator = MutateAction(config=llm_config)

    asyncio.run(mutate_operator.run(prompt=prompt))
    improved_prompt = mutate_operator.get_code_text()
    return improved_prompt


def llm_crossover(prompt_1, prompt_2):
    llm_config = Config.default()
    llm_config.llm.model = "gpt-4-turbo"
    llm_config.llm.temperature = 0.7
    crossover_operator = CrossoverAction(config=llm_config)

    asyncio.run(crossover_operator.run(prompt_1=prompt_1, prompt_2=prompt_2))
    improved_prompt = crossover_operator.get_code_text()
    return improved_prompt


class LLMEvaluator(object):
    def __init__(self, config, evaluator_dir):
        self.config = config
        self.evaluator_dir = evaluator_dir
        self.dummy_mode = self.config.get("dummy_mode", False)
        self.n_workers = self.config.get("n_workers", 1)
        assert self.n_workers > 0
        self.llm_model = self.config.get("llm_model", "gpt-3.5-turbo")
        self.restart_interval = self.config("restart_interval", 999)

        self.logger = logging.getLogger('evolve_role')
        self.reset()

    def reset(self):
        self.gen = 0
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); del self.pool
        self.pool = Pool(self.n_workers)

    def evaluate(self, population):
        self.gen += 1
        if self.gen % self.restart_interval == 0:
            self.reset()

        if n_workers == 1 or self.dummy_mode:
            result_dicts = []
            for indv in population:
                if self.dummy_mode:
                    fitness = random.random()
                else:
                    fitness = self._evalplus(indv.role, indv.id)
                result_dict = {}
                result_dict['fitness'] = fitness
                result_dict['true_fitness'] = fitness
                result_dicts.append(result_dict)
        else:
            result_dicts = self.pool.map(self._evalplus_wrapper, population)
        return result_dicts

    def _evalplus_wrapper(self, indv):
        fitness = self._evalplus(indv.role, indv.id)
        result_dict = {}
        result_dict['fitness'] = fitness
        result_dict['true_fitness'] = fitness
        return result_dict

    def _evalplus(self, prompt_template, eval_id, dataset='humaneval'):
        result_dir = os.path.join(
            self.evaluator_dir, "%s_ID-%s_T-%d" % (dataset, eval_id,
                time.time()))

        if dataset == 'humaneval':
            problems = get_human_eval_plus()
        else:
            assert dataset == 'mbpp'; problems = get_mbpp_plus()
        # results = []

        for task_id, problem in problems.items():
            prompt = problem['prompt']
            logger.info("\n\n#### Task ID: %s, Prompt:\n%s" % (task_id, prompt))

            team, coder = create_new_team(self.llm_model)
            coder.set_prompt_template(prompt_template)
            team.run_project(prompt)
            asyncio.run(team.run(n_round=1))
            output = coder.get_code_text()
            logger.info("#### MetaGPT Output:\n%s" % output)

            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
            with open(result_file, 'w') as f:
                f.write(output)
            # results.append({'task_id': task_id, 'solution': output})

        evalplus_fp = os.path.join(result_dir, "evalplus.txt")
        os.system("evalplus.evaluate --dataset %s --samples %s | tee %s"
            % (dataset, result_dir, evalplus_fp))
        time.sleep(0.25)
        with open(os.path.join(result_dir, "prompt_template.txt"), "w") as f:
            f.write(coder.get_prompt_template())

        return extract_evalplus_score(evalplus_fp, self.logger)


#### Unit tests ####
def _test_mutation_crossover():
    PROMPT_TEMPLATE_1 = '''
Write a python function that can {instruction}.
Return ```python your_code_here ``` with NO other texts,
your code:
'''
    PROMPT_TEMPLATE_2 = '''
### Task Description
Write a Python function that {instruction}. Ensure your code adheres to the following guidelines for quality and maintainability:

- **Modularity**: Break down the solution into smaller, reusable components where applicable.
- **Readability**: Use meaningful variable and function names that clearly indicate their purpose or the data they hold.

### Your Code
Return your solution in the following format:
```python your_code_here ```
with no additional text outside the code block.
'''
    output = llm_mutate(PROMPT_TEMPLATE_1)
    print("### LLM_MUTATE RETURN VALUE ###")
    print(output)

    output = llm_crossover(PROMPT_TEMPLATE_1, PROMPT_TEMPLATE_2)
    print("### LLM_CROSSOVER RETURN VALUE ###")
    print(output)


def _test_evaluator():
    from role_ga import Individual
    indv = Individual({}, gen_created=0)
    indv.role = \
'''
Write a python function that can {instruction}.
Return ```python your_code_here ``` with NO other texts,
your code:
'''
    population = [indv]
    evaluator = LLMEvaluator({}, evaluator_dir='.')
    result_dicts = evaluator.evaluate(population)
    print("Evaluation results:")
    print(result_dicts)


def _test_evalplus_extractor():
    score = extract_evalplus_score(
        "results/humaneval_ID-G-0_ID-KZJyETCnkrAI/evalplus.txt")
    print(score, type(score))

def _test_prompt_extractor():
    prompt_template = \
"""PROMPT_TEMPLATE: str = '''
### Task Description
Write a Python function that {instruction}. Ensure your code adheres to the following guidelines for quality and maintainability:

### Your Code
Return your solution in the following format:
```python
# Your code here
```
with no additional text outside the code block.
'''"""

    print("Original prompt template:")
    print(prompt_template)
    print("Extracted prompt template:")
    print(parse_prompt_template(prompt_template))

if __name__ == "__main__":
    _test_mutation_crossover()
