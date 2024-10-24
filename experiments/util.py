import asyncio
import calendar
import copy
from collections import defaultdict
import datetime
import functools
import json
import os
import math
import pickle
import platform
import pprint
import psutil
import pytz
import re
import shutil
import sys
import time
import traceback

import jsonpickle
import numpy as np
from pytz import timezone
from ruamel.yaml import YAML

from alg_util import is_numpy_type, randomword

OBJECTIVES = {'base_score': lambda x: x,
    'plus_score': lambda x: x,
    'hybrid_score': lambda x: x,
    'weighted_base_score': lambda x: x,
    'weighted_plus_score': lambda x: x,
    'weighted_hybrid_score': lambda x: x,
    'wall_time_sec': lambda x: -x,
    'user_time_sec': lambda x: -x,
    'sys_time_sec': lambda x: -x,
    'num_instructions': lambda x: -x,
    'memory_usage_mb': lambda x: -x}
SLEEP_TIME = 5


def delete_contents_in_directory(directory_path, verbose=False):
    with os.scandir(directory_path) as entries:
        for entry in entries:
            if verbose: print("Deleting %s" % entry.path)
            try:
                if entry.is_file(): os.unlink(entry.path)
                else: shutil.rmtree(entry.path)
            except:
                if verbose: print("Deletion error\n%s" % traceback.format_exc())


def clear_autogen_cache():
    os.system("rm -rf .cache >/dev/null 2>&1")
    delete_contents_in_directory("/tmp/")
    # os.system("rm -rf /tmp/* >/dev/null 2>&1")


def format_prompt(prompt, instruction):
    try:
        prompt = prompt.format(instruction=instruction)
    except:
        try: # If {instruction} not found, search for first braces
            special_word = prompt[prompt.find("{"):prompt.find("}")+1]
            prompt = prompt.replace(special_word, instruction)
        except: # Last resort, just use problem directly
            prompt = instruction
    return prompt


def calc_weighted_evalplus_score(result_dir,
    evalplus_weights,
    normalize=True,
    debug_weights=False):
    if isinstance(evalplus_weights, str):
        assert os.path.exists(evalplus_weights)
        with open(evalplus_weights, 'r') as f:
            evalplus_weights = json.load(f)
    else: assert isinstance(evalplus_weights, dict)

    eval_json = os.path.join(result_dir, 'eval_results.json')
    assert os.path.exists(eval_json)
    with open(eval_json, 'r') as f: eval_dict = json.load(f)

    base_score = 0.0; max_base_score = 0.0
    plus_score = 0.0; max_plus_score = 0.0
    for task_id, result in eval_dict['eval'].items():
        base_weight = evalplus_weights['base_weights'][task_id]
        plus_weight = evalplus_weights['plus_weights'][task_id]
        if debug_weights:
            base_weight = 1.0; plus_weight = 1.0

        max_base_score += base_weight; max_plus_score += plus_weight
        if result[0]['base_status'] == "pass": base_score += base_weight
        if result[0]['plus_status'] == "pass": plus_score += plus_weight

    if normalize:
        base_score /= max_base_score; plus_score /= max_plus_score
    return base_score, plus_score


def collect_stats_from_chat(result_dict, *args, **kwargs):
    # pprint.pprint(groupchat_messages, width=120); time.sleep(999999)
    if 'eval_stats' not in result_dict: result_dict['eval_stats'] = {}
    stats_dict = result_dict['eval_stats']

    if 'agent_chat_count' not in stats_dict:
        stats_dict['agent_chat_count'] = {}
    if 'agent_code_count' not in stats_dict:
        stats_dict['agent_code_count'] = {}
    if 'agent_chat_time' not in stats_dict:
        stats_dict['agent_chat_time'] = 0.0

    for message in kwargs.get('groupchat_messages', []):
        agent_name = message['name']
        if not agent_name.endswith("_Expert"):
            continue

        if agent_name not in stats_dict['agent_chat_count']:
            stats_dict['agent_chat_count'][agent_name] = 0
        if agent_name not in stats_dict['agent_code_count']:
            stats_dict['agent_code_count'][agent_name] = 0

        stats_dict['agent_chat_count'][agent_name] += 1
        code = parse_code2(message['content'])
        if code is not None:
            stats_dict['agent_code_count'][agent_name] += len(code)

    stats_dict['agent_chat_time'] += kwargs.get('time_elapsed', 0.0)
    # pprint.pprint(result_dict); time.sleep(999999)


def extract_code_from_chat(chat_result):
    code = ""
    result = parse_code2(chat_result.summary)
    if result is not None:
        code = result
    else:
        for msg_dict in chat_result.chat_history[::-1]:
            result = parse_code2(msg_dict['content'])
            if result is not None:
                code = result
    return code


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


def parse_code2(rsp):
    # print(rsp); time.sleep(999999)
    if rsp is None: return None
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None


def parse_prompt_template(rsp):
    pattern = r"PROMPT_TEMPLATE: str = '''(.*)'''"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    code_text = code_text.lstrip().rstrip()
    return code_text


def killtree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        print("Killing child: %s" % child)
        child.kill()

    if including_parent:
        parent.kill()


def extract_evalplus(result_file, logger=None):
    assert os.path.exists(result_file); result_dict = {}
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if "(base tests)" in line:
                score = float(lines[i+1].rstrip().rsplit()[1])
                assert 0.0 <= score <= 1.0
                result_dict['base_score'] = score
            if "(base + extra tests)" in line:
                score = float(lines[i+1].rstrip().rsplit()[1])
                assert 0.0 <= score <= 1.0
                result_dict['plus_score'] = score
            # Linux performance metrics
            if platform.system() == 'Linux':
                if "Maximum resident set size (kbytes)" in line:
                    result_dict['memory_usage_mb'] = float(line.split()[-1])/1e3
                if "Elapsed (wall clock) time" in line:
                    result_dict['wall_time_sec'] = time_to_sec(line.split()[-1])
                if "User time" in line:
                    result_dict['user_time_sec'] = float(line.split()[-1])
                if "System time" in line:
                    result_dict['sys_time_sec'] = float(line.split()[-1])
            else: # MacOS performance metrics
                if "peak memory footprint" in line:
                    result_dict['memory_usage_mb'] = float(line.split()[0])/1e6
                if "instructions retired" in line:
                    result_dict['num_instructions'] = float(line.split()[0])
                if "real" in line and "user" in line and "sys" in line:
                    result_dict['wall_time_sec'] = float(line.split()[0])
                    result_dict['user_time_sec'] = float(line.split()[2])
                    result_dict['sys_time_sec'] = float(line.split()[4])

        assert "base_score" in result_dict and "plus_score" in result_dict
        result_dict['hybrid_score'] = \
            0.5 * result_dict["base_score"] + 0.5 * result_dict["plus_score"]
    except:
        stack_trace = traceback.format_exc()
        with open(result_file + ".err", "w") as f: f.write(stack_trace)
        if logger is None:
            print("Evalplus extraction failed: %s" % result_file)
            print(stack_trace)
        else:
            logger.info("Evalplus extraction failed: %s" % result_file)
            logger.info(stack_trace)
    finally:
        return result_dict


def time_to_sec(time_str):
    """Get seconds from time."""
    fields = [float(x) for x in time_str.split(':')]
    if len(fields) == 2:
        m, s = fields; h = 0.0
    else:
        h, m, s = fields
    return h * 3600.0 + m * 60.0 + s


def unzip(x):
    return [list(x) for x in zip(*x)]


def get_time(date=True, space=True):
    '''Creates a nicely formated timestamp'''
    if date:
        date_str = "%Y-%m-%d %H:%M:%S"
    else:
        date_str = "%H:%M:%S"

    if not space:
        date_str = date_str.replace(":", "-").replace(" ", "_")

    return datetime.datetime.now(timezone('US/Pacific')).strftime(date_str)


def datetime_to_epoch(datetime_str, space=True):
    if space:
        date, time = datetime_str.split()
        h, _m, s = time.split(":")
    else:
        date, time = datetime_str.split("_")
        h, _m, s = time.split("-")
    y, m, d = date.split("-")
    t = datetime.datetime(int(y), int(m), int(d), int(h), int(_m), int(s))
    # tz = pytz.timezone('America/Los_Angeles'); t = t.astimezone(tz)
    return calendar.timegm(t.timetuple())


def sanitize_result_dict(result_dict):
    '''Converts numpy types to python built ins'''
    if type(result_dict) in [list, tuple]:
        return [sanitize_result_dict(x) for x in result_dict]
    elif type(result_dict) is dict:
        new_result_dict = {}
        for k in result_dict:
            new_result_dict[k] = sanitize_result_dict(result_dict[k])
        return new_result_dict
    elif is_numpy_type(result_dict):
        if type(result_dict) is np.ndarray:
            return result_dict.tolist()
        else:
            return result_dict.item()
    else:
        return result_dict


def get_result_from_cache(key, cache_dir='/tmp/'):
    try:
        cache_file = os.path.join(cache_dir, str(key) + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                result_dict = pickle.load(f)
            return result_dict
        else:
            return None
    except:
        traceback.print_exc()
        return None


def save_result_to_cache(key, result_dict, cache_dir='/tmp/'):
    tmp_file = os.path.join(cache_dir, randomword(24))
    cache_file = os.path.join(cache_dir, str(key) + '.pkl')
    assert not os.path.exists(cache_file)

    with open(tmp_file, 'wb') as f:
        pickle.dump(result_dict, f)
        f.flush()
        os.fsync(f.fileno())
        f.close()

    os.rename(tmp_file, cache_file)
    assert os.path.exists(cache_file)


def save_global_config(filepath, global_dict):
    def strip_python_tags(s):
        result = []
        for line in s.splitlines():
            idx = line.find("!!python/")
            if idx > -1:
                line = line[:idx]
            result.append(line)
        return '\n'.join(result)

    config_dict = {}
    for key, value in global_dict.items():
        # print(key, value)
        if key.isupper() and not key.startswith('__'):
            config_dict[key] = copy.deepcopy(value)

    # jsonpickle.set_encoder_options('simplejson',
    #     use_decimal=True, sort_keys=True)
    # jsonpickle.set_preferred_backend('simplejson')
    # s = jsonpickle.encode(config_dict, indent=4)
    with open(filepath, 'w') as f:
        # f.write(s)
        YAML().dump(config_dict, f)
    print("Dumped global config to file: %s" % filepath)


def get_indv_config(experiment_dir, config_name="config.yaml"):
    try:
        with open(os.path.join(experiment_dir, config_name), "r") as f:
            exp_cfg = YAML().load(f)
        indv_config = exp_cfg['role_ga_config']['indv_config']
        print("Indv config:"); pprint.pprint(indv_config)
    except:
        traceback.print_exc()
        print("Cannot load indv config!"); time.sleep(3); indv_config = {}
    return indv_config


def get_eval_config(experiment_dir, config_name="config.yaml"):
    try:
        with open(os.path.join(experiment_dir, config_name), "r") as f:
            exp_cfg = YAML().load(f)
        eval_config = exp_cfg['llm_evaluator_config']
        print("Evaluator config:"); pprint.pprint(eval_config)
    except:
        traceback.print_exc()
        print("Cannot load evaluator config!"); time.sleep(3); eval_config = {}
    return eval_config

