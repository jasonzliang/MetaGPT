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

from llm_evaluator import _setup_indv, _setup_evaluator
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
    'problem_list': ['1'],
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
    with open(code_file, 'r') as f:
        return f.read()


def _get_prompt(prob_id,
    num_steps,
    result_dir):
    sub_dir = _get_subdir(is_code=False)
    prompt_file = os.path.join(result_dir, sub_dir, f"{prob_id}.{num_steps}.py")
    assert os.path.exists(code_file)
    with open(prompt_file, 'r') as f:
        return f.read()


def self_improve_loop(main_role_fp=None,
    team_role_fp=None,
    evolve_mode="team",
    num_gen=50,
    init_seed=0,
    result_dir='results/self_improve_%s' % get_time(space=False),
    scicode=True):

    indv = _setup_indv(main_role_fp, team_role_fp, evolve_mode)
    _eval = _setup_evaluator(n_indv, result_dir, scicode)
    assert len(_eval.problem_list) == 1

    print(indv.main_role); print(indv.team_role)
    pprint.pprint(indv.llm_config); pprint.pprint(_eval.config)

    counter = init_seed
    for i in range(num_gen):
        child = indv.create_child(i)
        child._set_id(i, seed=counter)
        counter += 1; population = [child]
        result_dicts = _eval.evaluate(population); _eval.reset()
        assert len(result_dicts) == 0; result_dict = result_dicts[0]
        print("Evaluation results:"); pprint.pprint(result_dict)

        correct_dict = result_dict['scicode_result']['correct_dict']
        subprob_counts, prob_tests = _load_jsonl(_eval.dataset)
        prob_id = _eval.problem_list[0]; n_steps = subprob_counts[prob_id]
        subprob_acc = len(correct_dict[prob_id])/float(n_steps)
        prob_solved = subprob_acc == 1.0

        prompt = _get_prompt(prob_id, n_steps, result_dict['result_dir'])
        code = _get_code(prob_id, n_steps, result_dict['result_dir'])
        tests = prob_tests[prob_id]



if __name__ == "__main__":
    pass