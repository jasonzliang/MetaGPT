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

from autogen_agent_builder import AgentBuilder
from ruamel.yaml import YAML
from scicode.parse.parse import read_from_jsonl

from autogen_team import CONFIG_FILE_OR_ENV
from llm_evaluator import _setup_indv, _setup_evaluator
from llm_operators import DEFAULT_MAIN_ROLE_MIN
from util import extract_evalplus, extract_code_from_chat, killtree, get_time
from util import format_prompt, clear_autogen_cache, collect_stats_from_chat
from util import calc_weighted_evalplus_score
from util import EVALPLUS_OBJ, SCICODE_OBJ, SLEEP_TIME

EVALPLUS_EVAL_CONFIG = {
    'max_problems': 999,
    'dataset': 'humaneval',
}

SCICODE_EVAL_CONFIG = {
    'n_tries': 999,
    'max_problems': 999,
    'dataset': 'problems_dev',
    'with_background': False,
}

EVAL_LLM_CONFIG = {
    'model': 'gpt-4o'
}

EVAL_BUILDER_LLM_CONFIG = {
    'agent_model': 'gpt-4o',
    'builder_model': 'gpt-4o',
    'custom_coding_instruct': True,
    'max_round': 50,
}

EVAL_CHAT_LLM_CONFIG = {
    'model': 'gpt-4o'
}


def _get_scicode_problem_list(dataset=None, problem_order='complexity', whitelist=None):
    if dataset is None: dataset = SCICODE_EVAL_CONFIG['dataset']
    dataset_path = os.path.join("scicode_data", dataset + ".jsonl")
    data = read_from_jsonl(dataset_path); problem_list = []
    for problem in data:
        prob_id = problem['problem_id']
        if whitelist is None or prob_id in whitelist:
            problem_list.append(problem)

    if problem_order == 'random':
        random.shuffle(problem_list)
    elif problem_order == 'numeric':
        problem_list = sorted(problem_list, key=lambda x: int(x['problem_id']))
    else:
        assert problem_order == 'complexity'
        problem_list = sorted(problem_list, key=lambda x: len(x['sub_steps']))
    return [problem['problem_id'] for problem in problem_list]

def _get_subdir(is_code):
    if is_code: first_dir = "generated_code"
    else: first_dir = "prompt"
    with_background = SCICODE_EVAL_CONFIG['with_background']
    if with_background: third_dir = "with_background"
    else: third_dir = "without_background"
    return '%s/scicode_eval/%s' % (first_dir, third_dir)


def _load_jsonl(dataset):
    dataset_path = os.path.join("scicode_data", dataset + ".jsonl")
    assert os.path.exists(dataset_path)
    problems = read_from_jsonl(dataset_path)
    sub_steps = {}; general_tests = {}
    for problem in problems:
        prob_id = problem['problem_id']
        sub_steps[prob_id] = len(problem['sub_steps'])
        general_tests[prob_id] = problem['general_tests']
    return sub_steps, general_tests


def _get_output(prob_id,
    num_steps,
    result_dir,
    is_code=True):
    sub_dir = _get_subdir(is_code=is_code)
    output_file = os.path.join(result_dir, sub_dir, f"{prob_id}.{num_steps}.py")
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f: return f.read()


def _save_checkpoint(checkpoint_dict, result_dir):
    checkpoint_file = os.path.join(result_dir, "checkpoint.yaml")
    print("Saving checkpoint: %s" % checkpoint_file)
    with open(checkpoint_file, "w") as f:
        YAML().dump(checkpoint_dict, f)


def _load_checkpoint(result_dir):
    checkpoint_file = os.path.join(result_dir, "checkpoint.yaml")
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint: %s" % checkpoint_file)
        with open(checkpoint_file, "r") as f:
            return dict(YAML().load(f))
    else: return None


def _is_stuck(prob_id, history, threshold):
    if threshold is None or len(history) < threshold: return False
    prev_prob_ids = set(history[-threshold:])
    return len(prev_prob_ids) == 1 and prob_id in prev_prob_ids


# Todo:
# -Get error messages from failed test and use them to update agents
# -If stuck on problem, move onto next one and come back later (Done)
# -Reset team role to initial one if stuck on problem (Done)
# -Give ground truth code to agent if stuck on problem
# -Move self_improve_loop arguments/configuration into a dictionary

def self_improve_loop(team_role_fp=None,
    result_dir='results/self_improve_%s' % get_time(space=False),
    num_gen=100,
    init_seed=0,
    problem_list=_get_scicode_problem_list(),
    # problem_list=['1'],
    update_n_agents=None,
    update_teamwork=True,
    coding_instruct=True,
    solve_all=False,
    reset_team_role=False,
    stuck_threshold=10,
    scicode=True):

    if not scicode: raise Exception("Evalplus self-improve not implemented!")

    _eval = _setup_evaluator(1, result_dir, scicode, SCICODE_EVAL_CONFIG)
    EVAL_BUILDER_LLM_CONFIG['custom_coding_instruct'] = coding_instruct
    indv = _setup_indv(main_role_fp=DEFAULT_MAIN_ROLE_MIN,
        team_role_fp=team_role_fp,
        evolve_mode="team",
        builder_llm_config=EVAL_BUILDER_LLM_CONFIG,
        chat_llm_config=EVAL_CHAT_LLM_CONFIG,
        indv_llm_config=EVAL_LLM_CONFIG)
    init_team_role = indv.team_role
    pprint.pprint(indv.llm_config); pprint.pprint(_eval.config)

    curr_team_role = None; start_gen = 0; solved_problems = []; history = []
    checkpoint_dict = _load_checkpoint(result_dir)
    if checkpoint_dict is not None:
        curr_team_role = checkpoint_dict['curr_team_role']
        start_gen = checkpoint_dict['gen'] + 1
        init_seed = checkpoint_dict['init_seed']
        solved_problems = checkpoint_dict['solved_problems']
        history = checkpoint_dict['history']
        problem_list = [x for x in problem_list if x not in solved_problems]
    if len(problem_list) == 0:
        print("All problems solved, exiting self improve loop"); return

    _eval.problem_list = [problem_list.pop(0)]
    for i in range(start_gen, num_gen):
        prob_id = _eval.problem_list[0]
        indv._set_id(i, seed=i + init_seed, suffix='PROB-%s' % prob_id)
        if curr_team_role is not None: indv.team_role = curr_team_role

        print(indv.main_role); print(indv.team_role)

        result_dicts = _eval.evaluate([indv], gen=i)
        assert len(result_dicts) > 0; result_dict = result_dicts[0]
        eval_result_dir = result_dict['result_dir']
        print("Evaluation results:"); pprint.pprint(result_dict)

        sub_steps_dict, test_cases_dict = _load_jsonl(_eval.dataset)
        n_steps = sub_steps_dict[prob_id]
        # prompt = _get_output(prob_id, n_steps, eval_result_dir, is_code=False)
        code_generated = _get_output(prob_id, n_steps, eval_result_dir, is_code=True)
        test_cases = test_cases_dict[prob_id]

        solved_steps = result_dict['eval_result']['correct_dict'][prob_id]
        subprob_acc = len(solved_steps)/float(n_steps)
        final_step = "%s.%s" % (prob_id, n_steps)
        if solve_all: overall_acc = 1.0 if subprob_acc == 1.0 else 0.0
        else: overall_acc = 1.0 if final_step in solved_steps else 0.0

        code_performance = """Note: overall accuracy score is more important, focus on maximizing it.\nSub-problem accuracy score: %s\nOverall accuracy score: %s"""
        code_performance = code_performance % (subprob_acc, overall_acc)
        with open(os.path.join(eval_result_dir, "code_perf.txt"), 'w') as f:
            f.write(code_performance)

        builder_llm_config = indv.llm_config['builder_llm_config']
        builder = _eval.autogen_builder; assert builder is not None
        builder.update_agents(
            code_generated=code_generated,
            test_cases=test_cases,
            code_performance=code_performance,
            n_agents=update_n_agents,
            update_teamwork=update_teamwork)
        updated_team_fp = os.path.join(eval_result_dir, "team_role_update.json")
        builder.save(updated_team_fp); curr_team_role = builder.cached_configs

        history.append(prob_id); new_problem = False
        if overall_acc == 1.0:
            solved_problems.append(prob_id)
            _eval.problem_list = [problem_list.pop(0)]
            new_problem = True
        elif _is_stuck(prob_id, history, stuck_threshold):
            problem_list.append(prob_id)
            _eval.problem_list = [problem_list.pop(0)]
            new_problem = True
        if new_problem and reset_team_role: curr_team_role = init_team_role

        checkpoint_dict = {
            'curr_team_role': curr_team_role,
            'gen': i,
            'init_seed': init_seed,
            'solved_problems': solved_problems,
            'history': history,
            'cfg_update_teamwork': update_teamwork,
            'cfg_update_n_agents': str(update_n_agents),
            'cfg_coding_instruct': coding_instruct,
            'cfg_current_problem': _eval.problem_list[0],
            'cfg_solve_all': solve_all,
            'cfg_reset_team_role': reset_team_role,
            'cfg_stuck_threshold': stuck_threshold,
        }
        _save_checkpoint(checkpoint_dict, result_dir); _eval.reset()
        if len(problem_list) == 0:
            print("All problems solved, exiting self improve loop"); break


if __name__ == "__main__":
    # print(_get_scicode_problem_list())
    self_improve_loop(team_role_fp=sys.argv[1],
        result_dir=sys.argv[2],
        update_teamwork=True if "update_teamwork" in sys.argv[2].lower() else False,
        coding_instruct=True if "coding_instruct" in sys.argv[2].lower() else False)
