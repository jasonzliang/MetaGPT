import asyncio
import copy
import glob
import json
import logging
import os
import platform
import re
import subprocess
import sys
import pprint
import random
import traceback
import time
import tqdm

from metagpt.logs import logger as mlogger
from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool
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
from util import extract_evalplus, extract_code_from_chat, killtree, get_time
from util import format_prompt, clear_autogen_cache, collect_stats_from_chat
from util import calc_weighted_evalplus_score
from util import OBJECTIVES, SLEEP_TIME


class LLMEvaluator(object):
    def __init__(self, config, evaluator_dir):
        self.config = config
        self.evaluator_dir = evaluator_dir
        # self.logger maybe not necessary with mlogger?
        self.logger = logging.getLogger('evolve_role')

        self.n_workers = self.config.get("n_workers", 1)
        assert self.n_workers > 0
        # self.llm_config = self.config.get("llm_config", {})
        self.dataset = self.config.get("dataset", "humaneval")
        assert self.dataset in ['humaneval', 'mbpp']
        self.objective = self.config.get("objective", "base_score")
        assert self.objective in OBJECTIVES
        self.evalplus_weights = self.config.get("evalplus_weights", None)
        if self.evalplus_weights is not None:
            assert os.path.exists(self.evalplus_weights)
        self.sanitize = self.config.get("sanitize", True)
        self.restart_interval = self.config.get("restart_interval", sys.maxsize)
        assert self.restart_interval > 0
        self.max_round = self.config.get("max_round", 15)
        assert self.max_round > 0
        self.max_problems = self.config.get("max_problems", sys.maxsize)
        assert self.max_problems > 0
        self.use_timestamp = self.config.get("use_timestamp", False)
        self.n_tries = self.config.get("n_tries", 2)
        assert self.n_tries > 0
        self.max_failures = self.config.get("max_failures", 10)
        assert self.max_failures > 0

        self.debug_mode = self.config.get("debug_mode", False)
        self.reset()

    def reset(self):
        self.gen = None
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); self.pool.clear(); del self.pool
        self.pool = Pool(self.n_workers)
        os.makedirs(self.evaluator_dir, exist_ok=True)
        # self._reset_pbar()
        # self.pool = ParallelPool(self.n_workers)

    # def _reset_pbar(self):
    #     try: self.pbar.close()
    #     except: pass
    #     self.pbar = None

    def _check_eval_progress(self, n_indv):
        # f = open(os.path.join(self.evaluator_dir, "progress.txt"), "w")
        # if self.pbar is None:
        #     self.pbar = tqdm.tqdm(total=n_problems * n_indv, file=f)
        #     self.pbar.update(0)
        # self.pbar.n = count; self.pbar.refresh(); self.pbar.update(0); f.close()
        if self.dataset not in ['humaneval', 'mbpp']:
            return

        n_problems = 164 if self.dataset == "humaneval" else 1000
        total = n_problems * n_indv

        try:
            eval_dirs = os.path.join(self.evaluator_dir, "evalG-%s*" % self.gen)
            cmd = 'find %s -name 0.py | wc;find %s -name 0.py -size +0 | wc' % \
                (eval_dirs, eval_dirs)
            result = subprocess.check_output(cmd, shell=True, text=True)
            result = result.replace("\n", " "); c,_,_,c2,_,_ = result.split()
            c = min(int(c), total); c2 = min(int(c2), total)
            percent = c/float(total) * 100; empty = max(c - c2, 0)

            cmd = 'ls %s | grep -i ".err$" | wc' % eval_dirs
            result = subprocess.check_output(cmd, shell=True, text=True)
            n_errors, _, _ = result.split(); n_errors = int(n_errors)
        except:
            mlogger.info("_check_eval_progress failed")
            mlogger.info(traceback.format_exc()); return

        summary = "Gen: %s, Results: %s/%s, Progress: %.2f%%, Empty: %s, Errors: %s\n" % \
            (self.gen, c, total, percent, empty, n_errors)
        mlogger.info(summary)
        with open(os.path.join(self.evaluator_dir, "progress.txt"), "w") as f:
            f.write(summary)

    def evaluate(self, population, gen=0):
        if gen % self.restart_interval == 0: self.reset()
        self.gen = gen

        evolve_mode = [indv.evolve_mode for indv in population]
        assert len(set(evolve_mode)) == 1; evolve_mode = evolve_mode[0]
        if evolve_mode == "single":
            eval_func = self._eval_indv_main_role
        else:
            eval_func = self._eval_indv_team_role

        if self.n_workers == 1 or self.debug_mode:
            result_dicts = []
            for indv in population:
                if self.debug_mode:
                    fitness = random.random()
                    result_dict = {}
                    result_dict['fitness'] = fitness
                    result_dict['true_fitness'] = fitness
                else:
                    result_dict = eval_func(indv)
                result_dicts.append(result_dict)
        else:
            async_results = self.pool.amap(eval_func, population)
            # self._reset_pbar()
            while not async_results.ready():
                self._check_eval_progress(len(population))
                time.sleep(SLEEP_TIME)
            result_dicts = async_results.get()
        # killtree(os.getpid(), including_parent=False) # Prevent zombie process
        return result_dicts


    def _setup_result_dir(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        if self.use_timestamp: # For debugging purposes
            result_dir = os.path.join(self.evaluator_dir,
                "%s_%s_T-%s" % (self.dataset, eval_id, get_time(space=False)))
        else:
            result_dir = os.path.join(self.evaluator_dir,
                "evalG-%s_%s_%s" % (int(self.gen), self.dataset, eval_id))

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

        result_dict = {}; fail_flag = os.path.join(result_dir, "max_failures")
        n_failures = 0 if not os.path.exists(fail_flag) else self.max_failures
        for i, (task_id, problem) in enumerate(problems.items()):
            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")
            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                continue

            if i < self.max_problems and n_failures < self.max_failures:
                mlogger.info("\n\n#### Task ID: %s Prompt:\n%s" % \
                    (task_id, problem['prompt']))
                n_tries = self.n_tries; err_str = ""
                while n_tries > 0:
                    try:
                        output = eval_func(problem, result_dict); break
                    except:
                        stack_trace = traceback.format_exc()
                        mlogger.info("eval_func failed for %s" % task_id)
                        mlogger.info(stack_trace)
                        err_str += stack_trace + "\n"
                        output = ""; n_tries -= 1; time.sleep(5)

                        if n_tries == 0:
                            err_fp = os.path.join(result_dir, '%s_%s.err' % \
                                (os.getpid(), get_time(space=False)))
                            with open(err_fp, 'w') as f: f.write(err_str)
                            n_failures += 1


                mlogger.info("#### Evalplus Problem Output:\n%s" % output)
            else:
                output = ""

            with open(result_file, 'w') as f: f.write(output)

        if n_failures >= self.max_failures: os.system("touch %s" % fail_flag)
        return result_dict

    def _sanitize(self, result_dir):
        if not self.sanitize: return
        os.system("evalplus.sanitize --samples %s >/dev/null 2>&1" % result_dir)
        os.system("rsync -avz %s-sanitized/ %s >/dev/null 2>&1" % \
            (result_dir, result_dir))
        os.system("rm -rf %s-sanitized" % result_dir)

    def _get_evalplus_results(self, result_dir):
        flag = "-v" if platform.system() == 'Linux' else '-l' # Flag for MacOS
        evalplus_fp = os.path.join(result_dir, "evalplus.txt")
        os.system("/usr/bin/time %s evalplus.evaluate " \
            "--dataset %s --samples %s 2>&1 | tee %s" \
            % (flag, self.dataset, result_dir, evalplus_fp))

        evalplus_result = extract_evalplus(evalplus_fp, mlogger)
        if self.objective.startswith('weighted_'):
            weighted_base_score, weighted_plus_score = \
                calc_weighted_evalplus_score(result_dir, self.evalplus_weights)
            evalplus_result['weighted_base_score'] = weighted_base_score
            evalplus_result['weighted_plus_score'] = weighted_plus_score
            evalplus_result['weight_hybrid_score'] = \
                0.5 * weighted_base_score + 0.5 * weighted_plus_score
        assert self.objective in evalplus_result, str(evalplus_result)
        assert "base_score" in evalplus_result, str(evalplus_result)

        result_dict = {}; scaling_fn = OBJECTIVES[self.objective]
        result_dict['fitness'] = scaling_fn(evalplus_result[self.objective])
        result_dict['true_fitness'] = evalplus_result['base_score']
        result_dict['result_dir'] = result_dir
        # Needed for multirun_evalplus in analysis
        result_dict['evalplus_result'] = evalplus_result
        return result_dict

    def _eval_indv_team_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        # if indv.evolve_mode != "both": main_role = DEFAULT_MAIN_ROLE
        assert team_role is not None

        builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
        builder_llm_config.update(indv.llm_config.get("builder_llm_config", {}))
        chat_llm_config = copy.copy(CHAT_LLM_CONFIG)
        chat_llm_config.update(indv.llm_config.get("chat_llm_config", {}))
        mlogger.info("Indv: %s\nChat config: %s\nBuilder config: %s" % \
            (indv.id, chat_llm_config, builder_llm_config))

        agent_list, agent_configs, builder, builder_dict = \
            init_builder(building_task=None,
                work_dir='/tmp/build_%s' % randomword(ID_LENGTH),
                builder_dict=team_role,
                builder_llm_config=builder_llm_config,
                use_builder_dict=True,
                clear_cache=True)
        # for agent in agent_list: pprint.pprint(agent.__dict__); print("\n")

        # @retry(Exception, tries=-1, delay=1, max_delay=32, backoff=2)
        def eval_func(problem, result_dict):
            prompt = format_prompt(prompt=main_role,
                instruction=problem['prompt'])

            start_time = time.time()
            chat_result, groupchat_messages = start_task(
                execution_task=prompt,
                agent_list=agent_list,
                coding=agent_configs["coding"],
                chat_llm_config=chat_llm_config,
                max_round=self.max_round)
            time_elapsed = time.time() - start_time

            output = extract_code_from_chat(chat_result); assert len(output) > 0
            collect_stats_from_chat(result_dict,
                groupchat_messages=groupchat_messages,
                time_elapsed=time_elapsed)
            builder.clear_all_agents(recycle_endpoint=False)
            return output

        result_dir = self._setup_result_dir(indv)
        result_dict = self._run_evalplus(result_dir, eval_func)
        self._sanitize(result_dir)
        result_dict.update(self._get_evalplus_results(result_dir))
        return result_dict

    def _eval_indv_main_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id

        @retry(Exception, tries=3, delay=1, backoff=2, logger=mlogger)
        def _eval_prompt(prompt_template, prompt):
            team, coder = create_new_team(
                indv.llm_config.get('metagpt_llm_config', {}))
            coder.set_prompt_template(prompt_template)
            team.run_project(prompt)
            asyncio.run(team.run(n_round=1))
            output = coder.get_code_text()
            assert len(output) > 0
            return output

        def eval_func(problem, result_dict):
            return _eval_prompt(main_role, problem['prompt'])

        result_dir = self._setup_result_dir(indv)
        self._run_evalplus(result_dir, eval_func)
        self._sanitize(result_dir)
        return self._get_evalplus_results(result_dir)


#### Unit tests ####
def _test_evaluator(main_role_fp=None,
    team_role_fp=None,
    test_err=False,
    n_indv=1,
    num_gen=1,
    max_problems=1,
    max_round=20,
    llm_model='gpt-4o-mini'):

    clear_autogen_cache()
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

    if test_err: llm_model = 'N/A'
    builder_llm_config = copy.copy(BUILDER_LLM_CONFIG)
    builder_llm_config['agent_model'] = llm_model
    builder_llm_config['builder_model'] = llm_model
    chat_llm_config = copy.copy(CHAT_LLM_CONFIG)
    chat_llm_config['model'] = llm_model
    indv.llm_config = {'model': llm_model,
        'builder_llm_config': builder_llm_config,
        'chat_llm_config': chat_llm_config}
    print(indv.main_role); print(indv.team_role)
    pprint.pprint(indv.llm_config)

    eval_config = {'n_workers': n_indv,
        'debug_mode': False,
        'max_problems': max_problems,
        'max_round': max_round,
        'use_timestamp': False}

    evaluator = LLMEvaluator(eval_config, evaluator_dir='results/')
    for i in range(num_gen):
        # a = indv.create_child(0); b = indv.create_child(0)
        # a.id = "G-0_ID-DNrcOL1irjdO"; b.id = "G-0_ID-i6usMJsHu9xr"

        population = [indv.create_child(i) for j in range(n_indv)]
        result_dicts = evaluator.evaluate(population); evaluator.reset()
        print("Evaluation results:"); pprint.pprint(result_dicts)


def _test_evalplus_extractor(
    result_dir="results/humaneval_results_1712181961/evalplus.txt"):
    result_dict = extract_evalplus(result_dir)
    print(result_dict)


def _test_calc_weighted_evalplus_score(
    result_dir="results/multirole_baseline/evalG-0_humaneval_G-0_ID-mtrPn1oyR2xM",
    evalplus_weights="config/5_19_role_evo_weights.json"):

    evalplus_result = extract_evalplus(os.path.join(result_dir, "evalplus.txt"))
    pprint.pprint(evalplus_result)

    print(calc_weighted_evalplus_score(result_dir, evalplus_weights))
    with open(evalplus_weights, 'r') as f: weights_dict = json.load(f)
    print(calc_weighted_evalplus_score(result_dir, weights_dict))

    print(calc_weighted_evalplus_score(result_dir, weights_dict,
        normalize=False, debug_weights=True))
    print(calc_weighted_evalplus_score(result_dir, weights_dict,
        normalize=True, debug_weights=True))


def _test_check_eval_progress(
    evaluator_dir="results/8_19_multirole_coding_prompt/llm_evaluator",
    gen=17,
    n_indv=20):

    log = logging.getLogger("evolve_role"); log.debug("debug")

    evaluator = LLMEvaluator(config={}, evaluator_dir=evaluator_dir)
    evaluator.gen = gen
    evaluator._check_eval_progress(n_indv)


if __name__ == "__main__":
    _test_check_eval_progress()
    # _test_calc_weighted_evalplus_score(evalplus_weights="config/5_19_role_evo_weights.json")
    # _test_calc_weighted_evalplus_score(evalplus_weights="config/8_6_multirole_weights.json")
    # _test_evaluator(team_role_fp='config/autogen_builder_init.json')
