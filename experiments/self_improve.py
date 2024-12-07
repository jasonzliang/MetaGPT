#!/usr/bin/env python
import asyncio
import copy
from collections import defaultdict
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
import seaborn as sns

from scicode.parse.parse import read_from_jsonl
from autogen_agent_builder import AgentBuilder
from autogen_team import CONFIG_FILE_OR_ENV
from llm_evaluator import _setup_indv, _setup_evaluator
from llm_operators import DEFAULT_MAIN_ROLE_MIN
from util import extract_evalplus, extract_code_from_chat, killtree, get_time
from util import format_prompt, clear_autogen_cache, collect_stats_from_chat
from util import calc_weighted_evalplus_score, yaml_dump
from util import EVALPLUS_OBJ, SCICODE_OBJ, SLEEP_TIME

EVALPLUS_EVAL_CONFIG = {
    'max_problems': 999,
    'dataset': 'humaneval',
    'debug_mode': 0
}

SCICODE_EVAL_CONFIG = {
    'n_tries': 999,
    'max_problems': 999,
    'dataset': 'problems_dev',
    'with_background': False,
    'debug_mode': 0
}

EVAL_LLM_CONFIG = {
    'model': 'gpt-4o'
}

EVAL_BUILDER_LLM_CONFIG = {
    'agent_model': 'gpt-4o',
    'builder_model': 'gpt-4o',
    'custom_coding_instruct': True,
    'temperature': 0.9
}

EVAL_CHAT_LLM_CONFIG = {
    'model': 'gpt-4o',
    'temperature': 0.01,
    'use_llm_lingua': False,
    'max_round': 69,
}

class SolutionSet(object):
    def __init__(self, problem_list, problem_steps, stuck_threshold):
        self.problem_list = problem_list
        assert len(self.problem_list) > 0
        self.problem_steps = problem_steps
        assert len(self.problem_list) == len(self.problem_steps)
        self.stuck_threshold = stuck_threshold
        assert self.stuck_threshold > 0
        self.reset()

    def reset(self):
        self.history = []
        self.solutions = {}
        for prob_id, prob_steps in zip(self.problem_list, self.problem_steps):
            self.solutions[prob_id] = Solution(prob_id, prob_steps)

    def is_solved(self):
        return len(self.solved_problems()) == len(self.problem_list)

    def solved_problems(self):
        return [x.prob_id for x in self.solutions.values() if x.is_solved()]

    def stuck_problems(self):
        stuck_prob = [x for x in self.solutions.values() if x.is_stuck()]
        stuck_prob = sorted(stuck_prob, key=lambda x: x.gen_stuck)
        return [x.prob_id for x in stuck_prob]

    def unsolved_problems(self):
        return [x for x in self.problem_list if x not in self.solved_problems() and \
            x not in self.stuck_problems()]

    def get_problem(self, return_list=True):
        unsolved_problems = self.unsolved_problems()
        stuck_problems = self.stuck_problems()
        # print(unsolved_problems); print(stuck_problems); exit()
        if len(unsolved_problems) == 0:
            if len(stuck_problems) == 0:
                assert self.is_solved(); prob_id = None
            else:
                prob_id = stuck_problems[0]
                self.solutions[prob_id].gen_stuck = None
        else:
            prob_id = unsolved_problems[0]
        if return_list: prob_id = [prob_id]
        return prob_id

    def add_problem_result(self, prob_id, gen, success, steps_solved):
        assert prob_id in self.solutions; problem = self.solutions[prob_id]
        problem.add_record(gen, success, steps_solved)
        self.history.append(prob_id)
        if self.is_stuck(success, prob_id): problem.gen_stuck = gen

    def is_stuck(self, success, prob_id):
        if success or len(self.history) < self.stuck_threshold: return False
        prev_prob_ids = set(self.history[-self.stuck_threshold:])
        return len(prev_prob_ids) == 1 and prob_id in prev_prob_ids

    def serialize(self):
        solutions = [v.serialize() for v in self.solutions.values()]
        return {'history': self.history,
            'solutions': solutions,
            'solved_problems': self.solved_problems(),
            'unsolved_problems': self.unsolved_problems(),
            'stuck_problems': self.stuck_problems()}

    def deserialize(self, ss_dict):
        self.history = ss_dict.get('history', [])
        for prob_id, solution in self.solutions.items():
            for s_dict in ss_dict.get('solutions', []):
                assert 'prob_id' in s_dict
                if prob_id == s_dict.get('prob_id'):
                    solution.deserialize(s_dict)


class Solution(object):
    def __init__(self, prob_id, prob_steps):
        self.prob_id = prob_id
        self.prob_steps = prob_steps
        self.reset()

    def reset(self):
        self.gen_record = []
        self.gen_solved = None
        self.steps_solved = 0
        self.gen_stuck = None

    def is_stuck(self):
        return self.gen_stuck is not None

    def is_solved(self):
        return self.gen_solved is not None

    def add_record(self, gen, success, steps_solved):
        assert not self.is_solved(); assert steps_solved <= self.prob_steps
        self.gen_record.append(gen)
        if success:
            self.gen_solved = gen; self.steps_solved = steps_solved
        else:
            self.steps_solved = max(self.steps_solved, steps_solved)

    def get_stats(self):
        return {'prob_id': self.prob_id,
            'gen_solved': self.gen_solved,
            'gen_stuck': self.gen_stuck,
            'steps_solved': self.steps_solved,
            'prob_steps': self.prob_steps,
            'num_tries': len(self.gen_record)}

    def serialize(self):
        s_dict = self.get_stats()
        s_dict['gen_record'] = self.gen_record
        return s_dict

    def deserialize(self, s_dict):
        assert self.prob_id == s_dict.get('prob_id')
        self.gen_solved = s_dict.get('gen_solved')
        self.steps_solved = s_dict.get('steps_solved', 0)
        self.gen_stuck = s_dict.get('gen_stuck')
        self.gen_record = s_dict.get('gen_record', [])


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


def _get_subdir(first_dir):
    with_background = SCICODE_EVAL_CONFIG['with_background']
    if with_background: third_dir = "with_background"
    else: third_dir = "without_background"
    return os.path.join(first_dir, 'scicode_eval', third_dir)


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


def _get_file(prob_id,
    num_steps,
    result_dir,
    first_dir="generated_code",
    suffix=".py"):
    sub_dir = _get_subdir(first_dir)
    output_file = os.path.join(result_dir, sub_dir, f"{prob_id}.{num_steps}{suffix}")
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f: return f.read()


def _get_code(prob_id, num_steps, result_dir):
    return _get_file(prob_id, num_steps, result_dir)


def _get_test_output(prob_id, num_steps, result_dir):
    return _get_file(prob_id, num_steps, result_dir, first_dir="test_logs",
        suffix="_output.txt")


def _save_checkpoint(checkpoint_dict, result_dir):
    checkpoint_file = os.path.join(result_dir, "checkpoint.yaml")
    print("Saving checkpoint: %s" % checkpoint_file)
    yaml_dump(checkpoint_dict, checkpoint_file)


def _load_checkpoint(result_dir):
    checkpoint_file = os.path.join(result_dir, "checkpoint.yaml")
    if os.path.exists(checkpoint_file):
        print("Loading checkpoint: %s" % checkpoint_file)
        with open(checkpoint_file, "r") as f:
            return dict(YAML().load(f))
    else: return None


def _get_perf_feedback(prob_id, n_steps, solved_steps, eval_result_dir):
    subprob_acc = float(solved_steps)/float(n_steps)
    final_step = "%s.%s" % (prob_id, n_steps)
    prob_solved = True if subprob_acc == 1.0 else False

    if prob_solved:
        code_performance = "Code generated is correct, all test cases passed!"
    else:
        code_performance = \
"""Code generated is not correct, accuracy: %s
Stack trace/exception for test cases:\n%s""" % \
            (subprob_acc, _get_test_output(prob_id, n_steps, eval_result_dir))

    with open(os.path.join(eval_result_dir, "code_performance.txt"), 'w') as f:
        f.write(code_performance)
    return prob_solved, code_performance


# Todo:
# -If stuck on problem, move onto next one and come back later (Done)
# -Reset team role to initial one if stuck on problem (Done)
# -Get error messages from failed test and use them to update agents (Done)
# -Learn from solved problems and create agent knowledge pool (Done)
# -Collect stats regarding how long it takes to solve problem (Done)
# -Move all of the update agent prompts to a separate file (Done)
# -Analyze agent descriptions for solved problems and merge them together (Done)
# -Visualize stats (num gen to solve problem) as a bar graph (Done)
# -Compressing agent chat history with LLMLingua text compressor (Done)
# -Create a library of different expert agents to choose from (WIP)
# -Create a hierarchical team where there is a leader that delegates tasks (WIP)
# -Have multiple rounds of competition between agents, everyone give score to code
# -Let agents “cheat” by looking at the ground truth code
# -Change self_improve_loop in a class and arguments/configuration into yaml/dict

def self_improve_loop(team_role_fp=None,
    result_dir='results/self_improve_%s' % get_time(space=False),
    num_gen=200,
    init_seed=0,
    problem_list=_get_scicode_problem_list()[::-1],
    # problem_list=['44'],
    update_n_agents=None,
    update_teamwork=True,
    coding_instruct=True,
    reset_team_role=False,
    stuck_threshold=8,
    include_insights=True,
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

    steps_dict, test_cases_dict = _load_jsonl(_eval.dataset)
    problem_steps = [steps_dict[prob_id] for prob_id in problem_list]
    solution_set = SolutionSet(problem_list, problem_steps, stuck_threshold)

    curr_team_role = None; start_gen = 0
    checkpoint_dict = _load_checkpoint(result_dir)
    if checkpoint_dict is not None:
        curr_team_role = checkpoint_dict.get('curr_team_role', curr_team_role)
        start_gen = checkpoint_dict.get('gen', start_gen)
        init_seed = checkpoint_dict.get('init_seed', init_seed)
        solution_set.deserialize(checkpoint_dict.get('solution_set', {}))

    if solution_set.is_solved():
        print("All problems solved, not starting self-improve loop"); return

    _eval.problem_list = solution_set.get_problem(); end_loop = False
    for i in range(start_gen, num_gen):
        if end_loop: break

        prob_id = _eval.problem_list[0]
        indv._set_id(i, seed=i + init_seed, suffix='PROB-%s' % prob_id)
        if curr_team_role is not None: indv.team_role = curr_team_role

        print(indv.main_role); print(indv.team_role)

        result_dicts = _eval.evaluate([indv], gen=i)
        assert len(result_dicts) > 0; result_dict = result_dicts[0]
        eval_result_dir = result_dict['result_dir']
        print("Evaluation results:"); pprint.pprint(result_dict)

        n_steps = steps_dict[prob_id]; test_cases = test_cases_dict[prob_id]
        solved_steps = len(result_dict['eval_result']['correct_dict'][prob_id])
        # prompt = _get_output(prob_id, n_steps, eval_result_dir, is_code=False)
        code_generated = _get_code(prob_id, n_steps, eval_result_dir)
        problem_solved, code_performance = _get_perf_feedback(prob_id,
            n_steps, solved_steps, eval_result_dir)

        # builder_llm_config = indv.llm_config['builder_llm_config']
        builder = _eval.autogen_builder; assert builder is not None
        builder.update_agents(
            code_generated=code_generated,
            test_cases=test_cases,
            discover_insight=problem_solved,
            code_performance=code_performance,
            n_agents=update_n_agents,
            update_teamwork=update_teamwork)
        updated_team_fp = os.path.join(eval_result_dir, "team_role_update.json")
        builder.save(updated_team_fp)
        curr_team_role = copy.copy(builder.cached_configs)
        del curr_team_role['code_execution_config']

        solution_set.add_problem_result(prob_id, i, problem_solved, solved_steps)
        if not solution_set.is_solved():
            next_problem = solution_set.get_problem()
            if problem_solved:
                assert _eval.problem_list != next_problem
                if reset_team_role: curr_team_role = init_team_role
            _eval.problem_list = next_problem
        else:
            print("All problems solved, exiting loop at gen %s" % (i + 1))
            end_loop = True

        checkpoint_dict = {
            'curr_team_role': curr_team_role,
            'gen': i + 1,
            'init_seed': init_seed,
            'solution_set': solution_set.serialize(),
            'cfg_update_teamwork': update_teamwork,
            'cfg_update_n_agents': str(update_n_agents),
            'cfg_coding_instruct': coding_instruct,
            'cfg_reset_team_role': reset_team_role,
            'cfg_stuck_threshold': stuck_threshold,
            'cfg_include_insights': include_insights,
        }
        _save_checkpoint(checkpoint_dict, result_dir); _eval.reset()

    _merge_agents(indv=indv,
        evaluator=_eval,
        result_dir=result_dir,
        include_insights=include_insights)
    print("Self-improve loop finished")


def _merge_agents(indv,
    evaluator,
    result_dir,
    include_insights=True,
    merge_insights_with_desc=False,
    output_dir=None):

    assert os.path.exists(result_dir)
    if output_dir is None: output_dir = result_dir

    checkpoint_dict = _load_checkpoint(result_dir)
    assert checkpoint_dict is not None

    agent_configs_list = []
    for solution in checkpoint_dict['solution_set']['solutions']:
        if solution['gen_solved'] is None: continue
        team_role_file = glob.glob(os.path.join(result_dir,
            "evalG-%s_*" % solution['gen_solved'], "team_role.json"))

        assert len(team_role_file) == 1; team_role_file = team_role_file[0]
        print("Merging agent config from %s" % team_role_file)
        with open(team_role_file, 'r') as f:
            team_role = json.load(f)
            agent_configs_list.append(team_role['agent_configs'])
    if len(agent_configs_list) == 0: return

    if include_insights:
        agent_insights = [agent_config.get('insights') for agent_config in \
            checkpoint_dict['curr_team_role']['agent_configs']]
    else:
        agent_insights = None

    agent_list, agent_configs, builder, builder_dict = \
        evaluator._init_builder(indv.team_role,
            indv.llm_config['builder_llm_config'])
    assert builder is not None

    agent_library = builder.generate_agent_library(agent_configs_list,
        merge_insights_with_desc)
    with open(os.path.join(output_dir, "agent_library.json"), 'w') as f:
        json.dump(agent_library, f, indent=4)

    builder.merge_agents(agent_configs_list, agent_insights,
        merge_insights_with_desc)
    builder.save(os.path.join(output_dir, "merged_team_role.json"))


def visualize_performance(result_dirs,
    use_glob=True,
    key='num_tries',
    key_filter=('gen_solved', None, 0),
    out_dir='results/'):

    if type(result_dirs) is str: result_dirs = [result_dirs]
    if use_glob:
        _result_dirs = []
        for result_dir in result_dirs:
            _result_dirs.extend(glob.glob(result_dir))
        result_dirs = list(set(_result_dirs))
    assert len(result_dirs) > 0
    result_dirs = sorted(result_dirs, key=lambda x: os.path.basename(x))

    solution_dict = defaultdict(list); solved_counter = defaultdict(list)
    for result_dir in result_dirs:
        checkpoint_dict = _load_checkpoint(result_dir)
        assert checkpoint_dict is not None
        for solution in checkpoint_dict['solution_set']['solutions']:
            if solution['gen_solved'] is not None:
                name = os.path.basename(result_dir)
                solved_counter[name].append(solution['num_tries'])

            prob_id = solution['prob_id']; assert key in solution
            _key, _cond, _value = key_filter
            if solution[_key] == _cond:
                solution_dict[prob_id].append(_value)
            else:
                solution_dict[prob_id].append(solution[key])

    categories = []; values = []
    for _key, _values in solution_dict.items():
        categories.append(_key); values += _values
    num_probs = len(solution_dict); groups = []
    for result_dir in result_dirs:
        name = os.path.basename(result_dir)
        n_solved = len(solved_counter[name])
        avg_tries = "%.2f" % np.mean(solved_counter[name])
        groups.append(name + " (%s/%s, avg tries: %s)" % \
            (n_solved, num_probs, avg_tries))

    # Create Pandas DataFrame
    data = {
        'categories': np.repeat(categories, len(groups)),
        'groups': np.tile(groups, len(categories)),
        'values': values,
    }
    assert len(data['categories']) == len(data['groups']) == len(data['values'])
    df = pd.DataFrame(data)
    plt.figure(figsize=(10 + len(result_dirs), 6))
    ax = sns.barplot(x='categories', y='values', hue='groups', data=df)

    # Customize the plot
    for i in range(1, len(categories)):
        plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)
    plt.grid(True, axis='y', color='lightgray', alpha=0.5)
    plt.title('Comparison of Problem %s for Self-Improve Experiments' % key)
    plt.xlabel('Problem ID')
    plt.ylabel(key)
    plt.legend(title='Experiments', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot to file
    out_file = os.path.join(out_dir, "comp_%s_%s.png" % \
        (key, os.path.basename(result_dirs[0])))
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    visualize_performance(["results/12_5*",
        "results/11_29*",
        "results/self_improve_11_24/11_*no_update"])
    # if "lingua" in sys.argv[2]:
    #     EVAL_CHAT_LLM_CONFIG['use_llm_lingua'] = True
    # self_improve_loop(team_role_fp=sys.argv[1],
    #     result_dir=sys.argv[2],
    #     update_teamwork=True if "update_teamwork" in sys.argv[2].lower() else False,
    #     coding_instruct=True if "coding_instruct" in sys.argv[2].lower() else False)
