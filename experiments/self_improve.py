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


def _get_scicode_problem_list(dataset=None, shuffle=False, whitelist=None):
    if dataset is None: dataset = SCICODE_EVAL_CONFIG['dataset']
    dataset_path = os.path.join("scicode_data", dataset + ".jsonl")
    data = read_from_jsonl(input_path); problem_list = []
    for problem in data:
        prob_id = problem['problem_id']
        if whitelist is None or prob_id in whitelist:
            problem_list.append(problem['problem_id'])
    if shuffle:
        random.shuffle(problem_list); return problem_list
    else:
        return sorted(problem_list, key=lambda x: int(x))


def _get_subdir(is_code):
    if is_code: first_dir = "generated_code"
    else: first_dir = "prompt"
    without_background = SCICODE_EVAL_CONFIG['without_background']
    if without_background: third_dir = "without_background"
    else: third_dir = "with_background"
    return '%s/scicode_eval/%s' % (first_dir, third_dir)


def _load_jsonl(dataset):
    dataset_path = os.path.join("scicode_data", dataset + ".jsonl")
    assert os.path.exists(dataset_path)
    problems = read_from_jsonl(dataset_path)
    subprob_counts = {}; prob_tests = {}
    for prob_id in problems:
        subprob_counts[prob_id] = problems[prob_id]['subprob_count']
        prob_tests[prob_id] = problems[prob_id]['general_tests']
    return subprob_counts, prob_tests


def _get_code(prob_id,
    num_steps,
    result_dir):
    sub_dir = _get_subdir(is_code=True)
    code_file = os.path.join(result_dir, sub_dir, f"{prob_id}.{num_steps}.py")
    assert os.path.exists(code_file)
    with open(code_file, 'r') as f: return f.read()


def _get_prompt(prob_id,
    num_steps,
    result_dir):
    sub_dir = _get_subdir(is_code=False)
    prompt_file = os.path.join(result_dir, sub_dir, f"{prob_id}.{num_steps}.py")
    assert os.path.exists(code_file)
    with open(prompt_file, 'r') as f: return f.read()


def _save_checkpoint(result_dir, checkpoint_dict):
    checkpoint_file = os.path.join("result_dir", "checkpoint.yaml")
    print("Saving checkpoint: %s" % checkpoint_file)
    with open(checkpoint_file, "r") as f:
        YAML().dump(checkpoint_dict, f)


def _load_checkpoint(result_dir):
    checkpoint_file = os.path.join("result_dir", "checkpoint.yaml")
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint: %s" % checkpoint_file)
        with open(checkpoint_file, "r") as f:
            return dict(YAML().load(f))
    else: return None


def self_improve_loop(team_role_fp=None,
    num_gen=50,
    init_seed=0,
    problem_list=['1'],
    result_dir='results/self_improve_%s' % get_time(space=False),
    scicode=True):

    if not scicode: raise Exception("Evalplus self-improve not implemented!")

    _eval = _setup_evaluator(1, result_dir, scicode, SCICODE_EVAL_CONFIG)
    _eval.problem_list = [problem_list.pop(0)]
    indv = _setup_indv(main_role_fp=DEFAULT_MAIN_ROLE_MIN,
        team_role_fp=team_role_fp,
        evolve_mode="team",
        builder_llm_config=EVAL_BUILDER_LLM_CONFIG,
        chat_llm_config=EVAL_CHAT_LLM_CONFIG,
        indv_llm_config=EVAL_LLM_CONFIG)
    pprint.pprint(indv.llm_config); pprint.pprint(_eval.config)

    curr_team_role = None; start_gen = 0
    checkpoint_dict = _load_checkpoint(result_dir)
    if checkpoint_dict is not None:
        curr_team_role = checkpoint_dict['curr_team_role']
        start_gen = checkpoint_dict['gen'] + 1
        init_seed = checkpoint_dict['init_seed']
        problem_list = checkpoint_dict['problem_list']

    for i in range(start_gen, num_gen):
        if curr_team_role is not None: indv.team_role = curr_team_role
        prob_id = _eval.problem_list[0]
        indv._set_id(i, seed=i + init_seed, suffix='PROB-%s' % prob_id)
        print(indv.main_role); print(indv.team_role)

        population = [indv]
        result_dicts = _eval.evaluate(population); _eval.reset()
        assert len(result_dicts) > 0; result_dict = result_dicts[0]
        result_dir = result_dict['result_dir']
        print("Evaluation results:"); pprint.pprint(result_dict)

        correct_dict = result_dict['eval_result']['correct_dict']
        subprob_counts, prob_tests = _load_jsonl(_eval.dataset)
        n_steps = subprob_counts[prob_id]
        subprob_acc = len(correct_dict[prob_id])/float(n_steps)
        fullprob_acc = 1.0 if subprob_acc == 1.0 else 0.0

        if fullprob_acc == 1.0 and len(problem_list) == 0:
            print("All problems solved, exiting self improve loop"); break
        elif fullprob_acc == 1.0 and len(problem_list) > 0:
            _eval.problem_list = [problem_list.pop()]
            print("Problem %s solved, moving to next one" % prob_id); continue

        prompt = _get_prompt(prob_id, n_steps, result_dir)
        code_generated = _get_code(prob_id, n_steps, result_dir)
        test_cases = prob_tests[prob_id]
        code_performance = """
Note: overall accuracy score is more important, focus on maximizing it
Subproblem accuracy score: %s\nOverall accuracy score: %s"""
        code_performance = code_performance % (subprob_acc, fullprob_acc)

        builder_llm_config = indv.llm_config['builder_llm_config']
        builder = _eval.autogen_builder; assert builder is not None
        builder.update_agents(
            code_generated=code_generated,
            test_cases=test_cases,
            code_performance=code_performance,
            n_agents=None,
            update_teamwork=True)
        updated_team_fp = os.path.join(result_dir, "updated_team_role.json")
        builder.save(updated_team_fp); curr_team_role = builder.cached_configs

        checkpoint_dict = {
            'curr_team_role': curr_team_role,
            'gen': i,
            'init_seed': init_seed,
            'problem_list': problem_list,
        }
        _save_checkpoint(checkpoint_dict, result_dir)

if __name__ == "__main__":
    self_improve_loop(team_role_fp=sys.argv[1], result_dir=sys.argv[2])