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
from role_ga import DEFAULT_ROLE


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
        self.eval_mode = self.config.get("eval_mode", "single")
        assert self.eval_mode in ['single', 'team', 'both']

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
                elif self.eval_mode == "single":
                    result_dict = self._eval_indv_main_role(indv)

                else: # self.eval_mode in ["both", "team"]
                    result_dict = self._eval_indv_team_role(indv)

                result_dicts.append(result_dict)
        else:
            result_dicts = self.pool.map(self._eval_indv, population)
        return result_dicts

    def _setup_result_dir(self, eval_id)
        result_dir = os.path.join(self.evaluator_dir,
            "%s_%s_T-%d" % (self.dataset, eval_id, time.time()))
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "main_role.txt"), "w") as f:
            f.write(main_role)
        if os.path.exists(team_role):
            os.system("cp %s %s" % (team_role,
                os.path.join(result_dir, "team_role.json")))
        return result_dir

    def _run_evalplus(self, result_dir, eval_func):
        if self.dataset == 'humaneval':
            problems = get_human_eval_plus()
        else:
            assert self.dataset == 'mbpp'; problems = get_mbpp_plus()

        for task_id, problem in problems.items():
            mlogger.info("\n\n#### Task ID: %s Prompt:\n%s" % (task_id, prompt))

            try:
                output = eval_func(problem)
            except:
                mlogger.info(traceback.format_exc())
                output = ""
            mlogger.info("#### Evalplus Problem Output:\n%s" % output)

            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
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

    def _eval_indv_team_role(indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        if self.eval_mode == "team": main_role = DEFAULT_ROLE

        builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
        builder_llm_config.update(indv.llm_config.get("builder_llm_config", {}))
        chat_llm_config = copy.copy(CHAT_LLM_CONFIG)
        chat_llm_config.update(indv.llm_config.get("chat_llm_config", {}))

        # @retry(Exception, tries=3, delay=1, backoff=2, logger=self.logger)
        def eval_func(problem):
            agent_list, agent_configs, builder, builder_dict = \
                init_builder(building_task=None,
                    work_dir='/tmp',
                    builder_cfg=json.dumps(team_role),
                    builder_llm_config=builder_llm_config,
                    dict_out=True)

            chat_result = start_task(
                execution_task=main_role % problem['prompt'],
                agent_list=agent_list,
                coding=agent_configs["coding"],
                chat_llm_config=chat_llm_config)
            builder.clear_all_agents(recycle_endpoint=True)
            output = extract_code_from_chat(chat_result)
            assert len(output) > 0
            return output

        result_dir = self._setup_result_dir(eval_id)
        self._run_evalplus(result_dir, eval_func)
        self.sanitize(result_dir)
        return self._get_evalplus_results(result_dir)

    def _eval_indv_main_role(self, indv):
        main_role, eval_id = indv.main_role, indv.id

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

        result_dir = self._setup_result_dir(eval_id)
        self._run_evalplus(result_dir, eval_func)
        self.sanitize(result_dir)
        return self._get_evalplus_results(result_dir)


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
