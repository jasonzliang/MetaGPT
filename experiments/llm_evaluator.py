#!/usr/bin/env python
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

import gdown
from metagpt.logs import logger as mlogger
from pathos.pools import ProcessPool as Pool
# from multiprocessing import Pool
# from pathos.pp import ParallelPool
from retry import retry
# from wrapt_timeout_decorator import *
from ruamel.yaml import YAML

from evalplus.data.humaneval import get_human_eval_plus
from evalplus.data.mbpp import get_mbpp_plus
from evalplus.data import write_jsonl
from scicode.parse.parse import read_from_jsonl

from alg_util import MIN_FITNESS, EPSILON, ID_LENGTH, MIN_POP_SIZE
from alg_util import randomword
from autogen_team import BUILDER_LLM_CONFIG, CHAT_LLM_CONFIG, CAPTAIN_LLM_CONFIG
from autogen_team import init_builder, start_task, init_captain_agent
from llm_operators import DEFAULT_MAIN_ROLE, DEFAULT_MAIN_ROLE_V2
from llm_operators import DEFAULT_MAIN_ROLE_MIN
from llm_operators import create_new_team
from scicode_eval import DEFAULT_PROMPT_TEMPLATE, BACKGOUND_PROMPT_TEMPLATE
from scicode_eval import Gencode, test_code
from util import EVALPLUS_OBJ, SCICODE_OBJ, SLEEP_TIME
from util import extract_evalplus, extract_code_from_chat, killtree, get_time
from util import format_prompt, clear_autogen_cache, collect_stats_from_chat
from util import calc_weighted_evalplus_score, yaml_dump, recursive_update


class EvalPlusEvaluator(object):
    def __init__(self, config, evaluator_dir):
        self.config = config
        self.evaluator_dir = evaluator_dir
        # self.logger maybe not necessary with mlogger?
        self.logger = logging.getLogger('evolve_role')
        self.log_to_file = self.config.get("log_to_file", True)
        self.restart_interval = self.config.get("restart_interval", sys.maxsize)
        self.use_timestamp = self.config.get("use_timestamp", False)
        self.n_tries = self.config.get("n_tries", 2)
        assert self.n_tries > 0
        self.max_failures = self.config.get("max_failures", 10)
        assert self.max_failures > 0
        self.n_workers = self.config.get("n_workers", 1)
        assert self.n_workers > 0
        # self.max_round = self.config.get("max_round", 15)
        # assert self.max_round > 0
        self.max_problems = self.config.get("max_problems", sys.maxsize)
        assert self.max_problems > 0
        self.problem_list = self.config.get("problem_list", [])
        if len(self.problem_list) > 0: self.max_problems = sys.maxsize
        self.use_captain_agent = self.config.get("use_captain_agent", False)

        # 0 = Normal, 1 = Print more debug messages, 2 = 1 + Dummy fitness
        self.debug_mode = self.config.get("debug_mode", 0)
        if self.debug_mode:
            mlogger.warn("Warning, debug mode turned on: %s" % self.debug_mode)
            time.sleep(0.5)

        if type(self).__name__ == "EvalPlusEvaluator":
            self._init_evalplus()
            self.reset()

    def _init_evalplus(self):
        # Evalplus specific configuration
        self.dataset = self.config.get("dataset", "humaneval")
        assert self.dataset in ['humaneval', 'mbpp']
        self.objective = self.config.get("objective", "base_score")
        assert self.objective in EVALPLUS_OBJ
        self.evalplus_weights = self.config.get("evalplus_weights", None)
        if self.evalplus_weights is not None:
            assert os.path.exists(self.evalplus_weights)
        assert self.restart_interval > 0
        self.sanitize = self.config.get("sanitize", True)

    def reset(self):
        self.gen = None
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); self.pool.clear(); del self.pool
        self.pool = Pool(self.n_workers)
        # self.pool = ParallelPool(self.n_workers)
        os.makedirs(self.evaluator_dir, exist_ok=True)
        self.autogen_builder = None

    def _check_eval_progress(self, n_indv):
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

        if self.n_workers == 1 or self.debug_mode > 0:
            mlogger.info("Single worker/debug mode evaluation enabled")
            result_dicts = []
            for indv in population:
                if self.debug_mode == 2:
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
        if self.log_to_file:
            with open(os.path.join(result_dir, "main_role.txt"), "w") as f:
                f.write(main_role)
            if team_role is not None:
                with open(os.path.join(result_dir, "team_role.json"), "w") as f:
                    json.dump(team_role, f, indent=4)
            with open(os.path.join(result_dir, "llm_config.yaml"), 'w') as f:
                YAML().dump(indv.llm_config, f)
            with open(os.path.join(result_dir, 'eval_config.yaml'), 'w') as f:
                YAML().dump(self.config, f)
        return result_dir

    def _run_evalplus(self, result_dir, eval_func):
        if self.dataset == 'humaneval':
            problems = get_human_eval_plus()
        else: # self.dataset == 'mbpp'
            problems = get_mbpp_plus()

        result_dict = {'result_dir': result_dir}
        fail_flag = os.path.join(result_dir, "max_failures")
        n_failures = 0 if not os.path.exists(fail_flag) else self.max_failures
        for i, (task_id, problem) in enumerate(problems.items()):
            task_id_dir = os.path.join(result_dir, task_id.replace("/", "_"))
            os.makedirs(task_id_dir, exist_ok=True)
            result_file = os.path.join(task_id_dir, "0.py")

            if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
                continue
            if len(self.problem_list) > 0 and task_id not in self.problem_list:
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
                        output = ""; n_tries -= 1; time.sleep(1)

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

    def _get_evalplus_results(self, result_dict):
        result_dir = result_dict['result_dir']
        self._sanitize(result_dir)
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

        scaling_fn = EVALPLUS_OBJ[self.objective]
        result_dict['fitness'] = scaling_fn(evalplus_result[self.objective])
        result_dict['true_fitness'] = evalplus_result['base_score']
        # Needed for multirun_evalplus in analysis
        result_dict['eval_result'] = evalplus_result

        yaml_dump(result_dict, os.path.join(result_dir, "result_dict.yaml"))
        return result_dict

    def _init_builder(self,
        team_role,
        chat_llm_config,
        builder_llm_config,
        captain_llm_config):
        if self.use_captain_agent:
            captain_agent = init_captain_agent(
                chat_llm_config=chat_llm_config,
                captain_llm_config=captain_llm_config)
            return [captain_agent], None, None, None
        else:
            return init_builder(
                building_task=None,
                use_builder_dict=True,
                builder_dict=team_role,
                builder_llm_config=builder_llm_config,
                clear_cache=True,
                debug_mode=self.debug_mode > 0)

    def _setup_llm_configs(self, indv):
        builder_llm_config = copy.deepcopy(BUILDER_LLM_CONFIG)
        builder_llm_config.update(indv.llm_config.get("builder_llm_config", {}))
        chat_llm_config = copy.deepcopy(CHAT_LLM_CONFIG)
        chat_llm_config.update(indv.llm_config.get("chat_llm_config", {}))
        captain_llm_config = copy.deepcopy(CAPTAIN_LLM_CONFIG)
        captain_llm_config = recursive_update(captain_llm_config,
            indv.llm_config.get("captain_llm_config", {}))

        mlogger.info("Indv: %s\nChat config: %s\nBuilder config: %s\nCaptain config: %s" % \
            (indv.id, chat_llm_config, builder_llm_config, captain_llm_config))

        return chat_llm_config, builder_llm_config, captain_llm_config

    def _eval_indv_team_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        # if indv.evolve_mode != "both": main_role = DEFAULT_MAIN_ROLE
        assert team_role is not None

        chat_llm_config, builder_llm_config, captain_llm_config = \
            self._setup_llm_configs(indv)
        agent_list, _, builder, _ = \
            self._init_builder(team_role, chat_llm_config, builder_llm_config,
                captain_llm_config)
        self.autogen_builder = builder
        # for agent in agent_list: pprint.pprint(agent.__dict__); print("\n")

        # @retry(Exception, tries=-1, delay=1, max_delay=32, backoff=2)
        def eval_func(problem, result_dict):
            start_time = time.time()
            prompt = format_prompt(prompt=main_role, instruction=problem['prompt'])
            log_file = os.path.join(result_dict['result_dir'], "chat_logs",
                "%s_chat.yaml" % problem['task_id']) if self.log_to_file else None
            chat_result, groupchat_messages = start_task(
                execution_task=prompt,
                agent_list=agent_list,
                use_captain_agent=self.use_captain_agent,
                chat_llm_config=chat_llm_config,
                builder=self.autogen_builder,
                builder_llm_config=builder_llm_config,
                log_file=log_file)
            time_elapsed = time.time() - start_time

            output = extract_code_from_chat(chat_result); assert len(output) > 0
            collect_stats_from_chat(result_dict,
                groupchat_messages=groupchat_messages,
                time_elapsed=time_elapsed)
            if builder is not None:
                builder.clear_all_agents(recycle_endpoint=False)
            return output

        result_dir = self._setup_result_dir(indv)
        result_dict = self._run_evalplus(result_dir, eval_func)
        result_dict = self._get_evalplus_results(result_dict)
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
        result_dict = self._run_evalplus(result_dir, eval_func)
        return self._get_evalplus_results(result_dict)


class SciCodeEvaluator(EvalPlusEvaluator):
    def __init__(self, config, evaluator_dir):
        super().__init__(config, evaluator_dir)
        self._init_scicode()
        self._download_testdata()
        super().reset()

    def _init_scicode(self):
        # Scicode specific stuff
        self.dataset = self.config.get("dataset", "problems_all")
        assert self.dataset in ['problems_all', 'problems_dev', 'example']
        self.dev_set = self.dataset == "problems_dev"
        self.dataset_path = os.path.join("scicode_data",
            self.dataset + '.jsonl')
        assert os.path.exists(self.dataset_path)
        self.with_background = self.config.get("with_background", False)
        self.include_bg_comments = self.config.get("include_bg_comments", True)
        self.objective = self.config.get("objective", "problem_acc")
        assert self.objective in SCICODE_OBJ
        self.cleanup_code = self.config.get("cleanup_code", True)
        self.cleanup_code_final = self.config.get("cleanup_code_final", False)
        if self.cleanup_code_final: self.cleanup_code = False
        # self.shuffle_seed = self.config.get("shuffle_seed", None)

    def _download_testdata(self):
        url = 'https://drive.google.com/uc?id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW'
        output = os.path.join("scicode_data", "test_data.h5")
        if os.path.exists(output): return
        gdown.download(url, output, quiet=False)

    def _eval_indv_main_role(self, indv):
        raise "Not implemented"

    def _eval_indv_team_role(self, indv):
        main_role, team_role, eval_id = indv.main_role, indv.team_role, indv.id
        # if indv.evolve_mode != "both": main_role = DEFAULT_MAIN_ROLE
        assert team_role is not None

        chat_llm_config, builder_llm_config, captain_llm_config = \
            self._setup_llm_configs(indv)
        agent_list, _, builder, _ = \
            self._init_builder(team_role, chat_llm_config, builder_llm_config,
                captain_llm_config)
        self.autogen_builder = builder
        # for agent in agent_list: pprint.pprint(agent.__dict__); print("\n")

        # @retry(Exception, tries=-1, delay=1, max_delay=32, backoff=2)
        def eval_func(prob_id, prompt, result_dict):
            start_time = time.time()
            log_file = os.path.join(result_dict['result_dir'], "chat_logs",
                "%s_chat.yaml" % prob_id) if self.log_to_file else None
            chat_result, groupchat_messages = start_task(
                execution_task=prompt,
                agent_list=agent_list,
                use_captain_agent=self.use_captain_agent,
                chat_llm_config=chat_llm_config,
                builder=self.autogen_builder,
                builder_llm_config=builder_llm_config,
                code_library=result_dict['code_library'],
                imports=result_dict['imports'],
                log_file=log_file)
            time_elapsed = time.time() - start_time

            # There is another extract code function in scicode_eval
            # Either this one or the one in scicode_eval may be disabled
            output = extract_code_from_chat(chat_result); assert len(output) > 0
            collect_stats_from_chat(result_dict,
                groupchat_messages=groupchat_messages,
                time_elapsed=time_elapsed)
            if builder is not None:
                builder.clear_all_agents(recycle_endpoint=False)

            return output

        result_dir = self._setup_result_dir(indv)
        result_dict = self._run_scicode(result_dir, eval_func)
        result_dict = self._get_scicode_results(result_dict)
        return result_dict

    def _run_scicode(self, result_dir, eval_func):
        gcode = Gencode(
            model="scicode_eval",
            output_dir=os.path.join(result_dir, "generated_code"),
            prompt_dir=os.path.join(result_dir, "prompt"),
            with_background=self.with_background,
            include_bg_comments=self.include_bg_comments,
            llm_eval_func=eval_func,
        )
        prompt_template = BACKGOUND_PROMPT_TEMPLATE if \
            self.with_background else DEFAULT_PROMPT_TEMPLATE
        problems = read_from_jsonl(self.dataset_path)
        problems = sorted(problems, key=lambda x: int(x['problem_id']))
        # if self.shuffle_seed is None:
        # else: random.Random(self.shuffle_seed).shuffle(problems)

        result_dict = {'result_dir': result_dir}
        fail_flag = os.path.join(result_dir, "max_failures")
        n_failures = 0 if not os.path.exists(fail_flag) else self.max_failures
        for i, problem in enumerate(problems):
            task_id = problem['problem_id']; steps = len(problem['sub_steps'])

            if i >= self.max_problems or n_failures >= self.max_failures:
                continue
            if len(self.problem_list) > 0 and task_id not in self.problem_list:
                continue

            mlogger.info("\n\n#### Task ID: %s Problem:\n%s" % \
                (task_id, problem['problem_description_main']))
            n_tries = self.n_tries; err_str = ""; code_file = None
            while n_tries > 0:
                try:
                    for i in range(steps):
                        if (task_id == "13" and i == 5) or \
                            (task_id == "62" and i == 0) or \
                            (task_id == "76" and i == 2):
                            continue
                        code_file = gcode.generate_response_with_steps(
                            prob_data=problem,
                            num_steps=i+1,
                            tot_steps=steps,
                            prompt_template=prompt_template,
                            result_dict=result_dict)
                        if self.cleanup_code:
                            self.autogen_builder.cleanup_code(code_file)
                    break
                except:
                    stack_trace = traceback.format_exc()
                    mlogger.info("eval_func failed for %s" % task_id)
                    mlogger.info(stack_trace)
                    err_str += stack_trace + "\n"
                    output = ""; n_tries -= 1; time.sleep(1)

                    if n_tries == 0:
                        err_fp = os.path.join(result_dir, '%s_%s.err' % \
                            (os.getpid(), get_time(space=False)))
                        with open(err_fp, 'w') as f: f.write(err_str)
                        n_failures += 1

        if self.cleanup_code_final: self.autogen_builder.cleanup_code(code_file)
        if n_failures >= self.max_failures: os.system("touch %s" % fail_flag)
        return result_dict

    def _get_scicode_results(self, result_dict):
        result_dir = result_dict['result_dir']
        scicode_result = test_code(
            model_name="scicode_eval",
            code_dir=os.path.join(result_dir, "generated_code"),
            log_dir=os.path.join(result_dir, "test_logs"),
            output_dir=result_dir,
            jsonl_path=self.dataset_path,
            dev_set=self.dev_set,
            with_background=self.with_background)

        scaling_fn = SCICODE_OBJ[self.objective]
        result_dict['fitness'] = scaling_fn(scicode_result[self.objective])
        result_dict['true_fitness'] = scicode_result['problem_acc']
        # Needed for multirun_evalplus in analysis
        result_dict['eval_result'] = scicode_result

        with open(os.path.join(result_dir, "result_dict.yaml"), 'w') as f:
            YAML().dump(result_dict, f)
        return result_dict


#### Unit tests ####
LLM_MODEL = "gpt-4o-2024-11-20"
# LLM_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

EVALPLUS_EVAL_CONFIG = {
    'max_problems': 999,
    'dataset': 'humaneval',
    'debug_mode': 0
}

SCICODE_EVAL_CONFIG = {
    'n_tries': 3,
    'max_problems': 999,
    'dataset': 'problems_all',
    'with_background': False,
    'problem_list': ['2'],
    'cleanup_code': False,
    'include_bg_comments': True,
    'debug_mode': 0,
    'use_captain_agent': True,
}

EVAL_LLM_CONFIG = {
    'model': LLM_MODEL
}

EVAL_CHAT_LLM_CONFIG = {
    'model': LLM_MODEL,
    'max_round': 75,
    'temperature': 0.01,
    'use_llm_lingua': False,
}

EVAL_BUILDER_LLM_CONFIG = {
    'agent_model': LLM_MODEL,
    'builder_model': LLM_MODEL,
    'custom_coding_instruct': True,
    'use_agent_library': False,
    'agent_lib_include_coding_instruct': True,
    'agent_lib_include_insights': True,
    'temperature': 0.9
}

EVAL_CAPTAIN_LLM_CONFIG = {
    "nested_config": {
        "autobuild_init_config": {
            "builder_model": LLM_MODEL,
            "agent_model": LLM_MODEL,
        },
        "group_chat_config": {"max_round": 5},
        "max_turns": 5
    }
}

def _setup_evaluator(
    n_workers,
    eval_dir,
    scicode,
    eval_config=None):

    if scicode:
        if eval_config is None: eval_config = SCICODE_EVAL_CONFIG
        Evaluator = SciCodeEvaluator
    else:
        if eval_config is None: eval_config = EVALPLUS_EVAL_CONFIG
        Evaluator = EvalPlusEvaluator

    eval_config.update({'n_workers': n_workers, 'use_timestamp': False})
    return Evaluator(eval_config, evaluator_dir=eval_dir)


def _setup_indv(
    main_role_fp,
    team_role_fp,
    evolve_mode,
    builder_llm_config=EVAL_BUILDER_LLM_CONFIG,
    chat_llm_config=EVAL_CHAT_LLM_CONFIG,
    captain_llm_config=EVAL_CAPTAIN_LLM_CONFIG,
    indv_llm_config=EVAL_LLM_CONFIG,
    clear_cache=False):

    # Warning: do not clear cache if running multiple experiments at same time!
    if clear_cache: clear_autogen_cache()
    from role_ga import Individual
    indv = Individual({}, gen_created=0)
    assert indv.team_role is None
    indv.evolve_mode = evolve_mode

    if main_role_fp is not None:
        if os.path.exists(main_role_fp):
            with open(main_role_fp, "r") as f:
                indv.main_role = f.read()
        else:
            indv.main_role = main_role_fp
    else:
        indv.main_role = DEFAULT_MAIN_ROLE_V2
    if team_role_fp is not None:
        assert os.path.exists(team_role_fp)
        assert evolve_mode in ['both', 'team']
        with open(team_role_fp, "r") as f:
            indv.team_role = json.load(f)
    else:
        with open("config/autogen_team_init.json", "r") as f:
            indv.team_role = json.load(f)

    _builder_llm_config = copy.deepcopy(BUILDER_LLM_CONFIG)
    _builder_llm_config.update(builder_llm_config)
    _chat_llm_config = copy.deepcopy(CHAT_LLM_CONFIG)
    _chat_llm_config.update(chat_llm_config)
    _captain_llm_config = copy.deepcopy(CAPTAIN_LLM_CONFIG)
    _captain_llm_config = recursive_update(_captain_llm_config,
        captain_llm_config)

    indv.llm_config = copy.deepcopy(indv_llm_config)
    indv.llm_config['builder_llm_config'] = _builder_llm_config
    indv.llm_config['chat_llm_config'] = _chat_llm_config
    indv.llm_config['captain_llm_config'] = _captain_llm_config
    return indv


def test_evaluator(
    main_role_fp=None,
    team_role_fp=None,
    evolve_mode="team",
    n_indv=1,
    indv_id_seed=0,
    num_gen=1,
    eval_suffix=get_time(space=False),
    scicode=True):

    assert eval_suffix is not None and len(eval_suffix) > 0
    if scicode: main_role_fp = DEFAULT_MAIN_ROLE_MIN
    _eval = _setup_evaluator(n_indv, "results/", scicode)
    indv = _setup_indv(main_role_fp, team_role_fp, evolve_mode)

    print(indv.main_role); print(indv.team_role)
    pprint.pprint(indv.llm_config); pprint.pprint(_eval.config)

    counter = 0
    for i in range(num_gen):
        population = []
        for j in range(n_indv):
            child = indv.create_child(i)
            if indv_id_seed is not None:
                child._set_id(i, seed=indv_id_seed + counter,
                    suffix=eval_suffix)
            else:
                child._set_id(i, seed=None, suffix=eval_suffix)
            counter += 1; population.append(child)

        result_dicts = _eval.evaluate(population, gen=i); _eval.reset()
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

    evaluator = EvalPlusEvaluator(config={}, evaluator_dir=evaluator_dir)
    evaluator.gen = gen
    evaluator._check_eval_progress(n_indv)


if __name__ == "__main__":
    if 'lingua' in sys.argv[2].lower():
        EVAL_CHAT_LLM_CONFIG['use_llm_lingua'] = True
    if 'library' in sys.argv[2].lower():
        EVAL_BUILDER_LLM_CONFIG['use_agent_library'] = True
    if 'background' in sys.argv[2].lower():
        SCICODE_EVAL_CONFIG['with_background'] = True
    if 'cleanup' in sys.argv[2]:
        SCICODE_EVAL_CONFIG['cleanup_code'] = True
    test_evaluator(team_role_fp=sys.argv[1], eval_suffix=sys.argv[2])
    # _test_calc_weighted_evalplus_score(evalplus_weights="config/5_19_role_evo_weights.json")
    # _test_calc_weighted_evalplus_score(evalplus_weights="config/8_6_multirole_weights.json")
    # _test_evaluator(team_role_fp='config/autogen_team3_init.json')
