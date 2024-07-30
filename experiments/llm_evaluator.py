import asyncio
import copy
import glob
import json
import logging
import os
import platform
import re
import sys
import pprint
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
# from pathos.pp import ParallelPool
from retry import retry
# from wrapt_timeout_decorator import *

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl

from alg_util import randomword
from alg_util import MIN_FITNESS, EPSILON, ID_LENGTH, MIN_POP_SIZE
from autogen_builder import init_builder, start_task
from autogen_builder import BUILDER_LLM_CONFIG, CHAT_LLM_CONFIG
from llm_operators import create_new_team
from llm_operators import DEFAULT_MAIN_ROLE
from util import extract_evalplus, extract_code_from_chat, killtree
from util import OBJECTIVES


class LLMEvaluator(object):
    def __init__(self, config, evaluator_dir):
        self.config = config
        self.evaluator_dir = evaluator_dir
        self.dummy_mode = self.config.get("dummy_mode", False)
        self.max_problems = self.config.get("max_problems", 9999)
        assert self.max_problems > 0
        self.n_workers = self.config.get("n_workers", 1)
        assert self.n_workers > 0
        self.llm_config = self.config.get("llm_config", {})
        self.dataset = self.config.get("dataset", "humaneval")
        assert self.dataset in ['humaneval', 'mbpp']
        self.objective = self.config.get("objective", "base_score")
        assert self.objective in OBJECTIVES
        self.sanitize = self.config.get("sanitize", True)
        self.restart_interval = self.config.get("restart_interval", 999)
        assert self.restart_interval > 0
        self.max_round = self.config.get("max_round", 15)
        assert self.max_round > 0
        # self.eval_mode = self.config.get("eval_mode", "single")
        # assert self.eval_mode in ['single', 'team', 'both']
        self.debug_no_timestamp = self.config.get("debug_no_timestamp", False)

        self.logger = logging.getLogger('evolve_role')
        self.reset()

    def reset(self):
        self.gen = 0
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); self.pool.clear()
        self.pool = ProcessPool(self.n_workers)
        # self.pool = ParallelPool(self.n_workers)

    def evaluate(self, population):
        self.gen += 1
        if self.gen % self.restart_interval == 0:
            self.reset()

        evolve_mode = population[0].evolve_mode
        if evolve_mode == "single":
            eval_func = self._eval_indv_main_role
        else:
            eval_func = self._eval_indv_team_role

        if self.n_workers == 1 or self.dummy_mode:
            result_dicts = []
            for indv in population:
                if self.dummy_mode:
                    fitness = random.random()
                    result_dict = {}
                    result_dict['fitness'] = fitness
                    result_dict['true_fitness'] = fitness
                else:
                    result_dict = eval_func(indv)
                result_dicts.append(result_dict)
        else:
            result_dicts = self.pool.map(eval_func, population)

        # killtree(os.getpid(), including_parent=False) # Prevent zombie process
        return result_dicts

    def _setup_result_dir(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        if self.debug_no_timestamp: # For debugging purposes
            result_dir = os.path.join(self.evaluator_dir,
                "%s_%s" % (self.dataset, eval_id))
        else:
            result_dir = os.path.join(self.evaluator_dir,
                "%s_%s_T-%d" % (self.dataset, eval_id, time.time()))

        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "main_role.txt"), "w") as f:
            f.write(main_role)
        if team_role is not None:
            with open(os.path.join(result_dir, "team_role.json"), "w") as f:
                json.dump(team_role, f, indent=4)
        return result_dir

    def _run_evalplus(self, result_dir, eval_func):
        if self.dataset == 'humaneval':
            problems = get_human_eval_plus()
        else: # self.dataset == 'mbpp'
            problems = get_mbpp_plus()

        for i, (task_id, problem) in enumerate(problems.items()):
            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                continue

            if i < self.max_problems:
                mlogger.info("\n\n#### Task ID: %s Prompt:\n%s" % \
                    (task_id, problem['prompt']))
                try:
                    output = eval_func(problem)
                except:
                    mlogger.info(traceback.format_exc())
                    output = ""
                mlogger.info("#### Evalplus Problem Output:\n%s" % output)
            else:
                output = ""

            with open(result_file, 'w') as f:
                f.write(output)

    def _sanitize(self, result_dir):
        if not self.sanitize: return
        os.system("evalplus.sanitize --samples %s >/dev/null" % result_dir)
        os.system("rsync -avz %s-sanitized/ %s >/dev/null" % \
            (result_dir, result_dir))
        os.system("rm -rf %s-sanitized" % result_dir)

    def _get_evalplus_results(self, result_dir):
        flag = "-v" if platform.system() == 'Linux' else '-l' # Flag for MacOS
        evalplus_fp = os.path.join(result_dir, "evalplus.txt")
        os.system("/usr/bin/time %s evalplus.evaluate " \
            "--dataset %s --samples %s 2>&1 | tee %s" \
            % (flag, self.dataset, result_dir, evalplus_fp))
        evalplus_result = extract_evalplus(evalplus_fp, self.logger)

        assert self.objective in evalplus_result
        assert "base_score" in evalplus_result
        scaling_fn = OBJECTIVES[self.objective]

        result_dict = {}
        result_dict['fitness'] = scaling_fn(evalplus_result[self.objective])
        result_dict['true_fitness'] = evalplus_result["base_score"]
        result_dict['result_dir'] = result_dir
        result_dict['evalplus_result'] = evalplus_result
        return result_dict

    def _eval_indv_team_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        if indv.evolve_mode != "both": main_role = DEFAULT_MAIN_ROLE
        assert team_role is not None

        builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
        builder_llm_config.update(indv.llm_config.get("builder_llm_config", {}))
        chat_llm_config = copy.copy(CHAT_LLM_CONFIG)
        chat_llm_config.update(indv.llm_config.get("chat_llm_config", {}))

        agent_list, agent_configs, builder, builder_dict = \
            init_builder(building_task=None,
                work_dir='/tmp/build_%s' % randomword(ID_LENGTH),
                builder_dict=team_role,
                builder_llm_config=builder_llm_config,
                use_builder_dict=True,
                clear_cache=True)

        # @retry(Exception, tries=3, delay=1, backoff=2, logger=self.logger)
        # @timeout(5, timeout_exception=TimeoutError, use_signals=True)
        def eval_func(problem):
            prompt = main_role
            try:
                prompt = prompt.format(instruction=problem['prompt'])
            except:
                # If {instruction} not found, search for first pair of braces
                special_word = prompt[prompt.find("{"):prompt.find("}")+1]
                prompt = prompt.replace(special_word, problem['prompt'])

            chat_result = start_task(
                execution_task=prompt,
                agent_list=agent_list,
                coding=agent_configs["coding"],
                chat_llm_config=chat_llm_config,
                max_round=self.max_round)

            builder.clear_all_agents(recycle_endpoint=False)
            output = extract_code_from_chat(chat_result)
            assert len(output) > 0
            return output
        result_dir = self._setup_result_dir(indv)

        self._run_evalplus(result_dir, eval_func)
        self._sanitize(result_dir)
        result_dict = self._get_evalplus_results(result_dir)
        return result_dict

    def _eval_indv_main_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id

        @retry(Exception, tries=3, delay=1, backoff=2, logger=self.logger)
        def _eval_prompt(prompt_template, prompt):
            team, coder = create_new_team(self.llm_config)
            coder.set_prompt_template(prompt_template)
            team.run_project(prompt)
            asyncio.run(team.run(n_round=1))
            output = coder.get_code_text()
            assert len(output) > 0
            return output

        def eval_func(problem):
            return _eval_prompt(main_role, problem['prompt'])

        result_dir = self._setup_result_dir(indv)
        self._run_evalplus(result_dir, eval_func)
        self._sanitize(result_dir)
        return self._get_evalplus_results(result_dir)


#### Unit tests ####
def _test_evaluator(main_role_fp=None, team_role_fp=None, test_err=False,
    max_problems=999, max_round=15, num_gen=2):
    from role_ga import Individual
    indv = Individual({}, gen_created=0)
    assert indv.team_role is None
    indv.evolve_mode = "single"

    if main_role_fp is not None:
        assert os.path.exists(main_role_fp)
        with open(main_role_fp, "r") as f:
            indv.main_role = f.read()
    else:
        indv.main_role = DEFAULT_MAIN_ROLE
    if team_role_fp is not None:
        indv.evolve_mode = "both"
        assert os.path.exists(team_role_fp)
        with open(team_role_fp, "r") as f:
            indv.team_role = json.load(f)

    llm_model = 'N/A' if test_err else 'gpt-3.5-turbo'
    builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
    builder_llm_config['model'] = llm_model
    chat_llm_config = copy.copy(CHAT_LLM_CONFIG)
    chat_llm_config['model'] = llm_model
    indv.llm_config = {'model': llm_model,
        'builder_llm_config': builder_llm_config,
        'chat_llm_config': chat_llm_config}
    print(indv.main_role); print(indv.team_role)
    pprint.pprint(indv.llm_config)

    eval_config = {'n_workers': 2,
        'dummy_mode': False,
        'max_problems': max_problems,
        'max_round': max_round,
        'debug_no_timestamp': True}

    evaluator = LLMEvaluator(eval_config, evaluator_dir='results/')
    for i in range(num_gen):
        a = indv.create_child(0); b = indv.create_child(0)
        a.id = "G-0_ID-DNrcOL1irjdO"; b.id = "G-0_ID-i6usMJsHu9xr"
        result_dicts = evaluator.evaluate([a, b])
        # evaluator.reset()

        print("Evaluation results:")
        print(result_dicts)


def _test_evalplus_extractor(
    result_dir="results/humaneval_results_1712181961/evalplus.txt"):
    result_dict = extract_evalplus(result_dir)
    print(result_dict)


def _test_parallel_eval(n=10):
    from role_ga import Individual
    population = [Individual({}, gen_created=0) for i in range(n)]
    for indv in population:
        indv.main_role = DEFAULT_MAIN_ROLE
    print(indv.main_role)

    eval_config = {'n_workers': n, 'dummy_mode': False}
    evaluator = LLMEvaluator(eval_config, evaluator_dir='results/')
    result_dicts = evaluator.evaluate(population)
    print("Evaluation results:")
    print(result_dicts)


if __name__ == "__main__":
    _test_evaluator(team_role_fp='autogen_builder_cfg.json', test_err=False)
