import asyncio
import copy
import glob
import json
import logging
import os
import platform
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

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl

from util import extract_evalplus, OBJECTIVES


class LLMEvaluator(object):
    def __init__(self, config, evaluator_dir):
        self.config = config
        self.evaluator_dir = evaluator_dir
        self.dummy_mode = self.config.get("dummy_mode", False)
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
        self.eval_method = self.config.get("eval_method", "single")
        assert self.eval_method in ['single', 'team']

        self.logger = logging.getLogger('evolve_role')
        self.reset()

    def reset(self):
        self.gen = 0
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); self.pool.clear()
        self.pool = ProcessPool(self.n_workers)

    def evaluate(self, population):
        self.gen += 1
        if self.gen % self.restart_interval == 0:
            self.reset()

        if self.n_workers == 1 or self.dummy_mode:
            result_dicts = []
            for indv in population:
                if self.dummy_mode:
                    fitness = random.random()
                    result_dict = {}
                    result_dict['fitness'] = fitness
                    result_dict['true_fitness'] = fitness
                elif self.eval_method == "single":
                    result_dict = self._eval_indv_main_role(indv)
                else: # self.eval_method == "team"
                    result_dict = self._eval_indv_team_role(indv)
                result_dicts.append(result_dict)
        else:
            result_dicts = self.pool.map(self._eval_indv, population)
        return result_dicts

    def _setup_result_dir(self, indv)
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        result_dir = os.path.join(self.evaluator_dir,
            "%s_%s_T-%d" % (self.dataset, eval_id, time.time()))
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "main_role.txt"), "w") as f:
            f.write(main_role)
        if os.path.exists(team_role):
            os.system("cp %s %s" % (team_role,
                os.path.join(result_dir, "team_role.json")))

        return result_dir

    def _eval_indv_team_role(indv):
        result_dir = self._setup_result_dir()
        if self.dataset == 'humaneval':
            problems = get_human_eval_plus()
        else:
            assert self.dataset == 'mbpp'; problems = get_mbpp_plus()

        for task_id, problem in problems.items():
            prompt = problem['prompt']
            mlogger.info("\n\n#### Task ID: %s Prompt:\n%s" % (task_id, prompt))

    def _eval_indv_main_role(self, indv):
        @retry(Exception, tries=5, delay=1, backoff=2, logger=self.logger)
        def _eval_prompt(prompt_template, prompt):
            team, coder = create_new_team(self.llm_config)
            coder.set_prompt_template(prompt_template)
            team.run_project(prompt)
            asyncio.run(team.run(n_round=1))
            output = coder.get_code_text()
            assert len(output) > 0
            return output

        result_dir = self._setup_result_dir()
        if self.dataset == 'humaneval':
            problems = get_human_eval_plus()
        else:
            assert self.dataset == 'mbpp'; problems = get_mbpp_plus()

        for task_id, problem in problems.items():
            prompt = problem['prompt']
            mlogger.info("\n\n#### Task ID: %s Prompt:\n%s" % (task_id, prompt))
            try:
                output = _eval_prompt(prompt_template, prompt)
            except:
                mlogger.info(traceback.format_exc())
                output = ""
            mlogger.info("#### MetaGPT Output:\n%s" % output)

            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
            with open(result_file, 'w') as f:
                f.write(output)

        self.sanitize()
        return self._run_evalplus(result_dir)

    def _sanitize(self, result_dir):
        if not self.sanitize: return
        os.system("evalplus.sanitize --samples %s >/dev/null" % result_dir)
        os.system("rsync -avz %s-sanitized/ %s >/dev/null" % \
            (result_dir, result_dir))
        os.system("rm -rf %s-sanitized" % result_dir)

    def _run_evalplus(self, result_dir):
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


#### Unit tests ####
def _test_evaluator(prompt_fp=None, test_err=False):
    from role_ga import Individual
    indv = Individual({}, gen_created=0)
    if prompt_fp is not None and os.path.exists(prompt_fp):
        with open(prompt_fp, "r") as f:
            indv.role = f.read()
    else:
        indv.role = PROMPT_TEMPLATE_1

    print(indv.role); population = [indv]
    llm_model = 'N/A' if test_err else 'gpt-3.5-turbo'
    eval_config = {'n_workers': 1, 'dummy_mode': False,
        'llm_config': {'model': llm_model}}
    evaluator = LLMEvaluator(eval_config, evaluator_dir='results/')
    result_dicts = evaluator.evaluate(population)
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
        indv.role = PROMPT_TEMPLATE_1
    print(indv.role)

    eval_config = {'n_workers': n, 'dummy_mode': False}
    evaluator = LLMEvaluator(eval_config, evaluator_dir='results/')
    result_dicts = evaluator.evaluate(population)
    print("Evaluation results:")
    print(result_dicts)


if __name__ == "__main__":
    _test_evaluator(test_err=False)
