import asyncio
import copy
import datetime
import functools
import os
import math
import pickle
import psutil
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
    'wall_time_sec': lambda x: -x,
    'user_time_sec': lambda x: -x,
    'sys_time_sec': lambda x: -x,
    'num_instructions': lambda x: -x,
    'memory_usage_mb': lambda x: -x}


def parse_code(rsp):
    pattern = r"```python(.*)```"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    return code_text


def parse_prompt_template(rsp):
    pattern = r"PROMPT_TEMPLATE: str = '''(.*)'''"
    match = re.search(pattern, rsp, re.DOTALL)
    code_text = match.group(1) if match else rsp
    code_text = code_text.lstrip().rstrip()
    return code_text


def killtree(pid, including_parent=True):
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        print "child", child
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
            if "Maximum resident set size (kbytes)" in line:
                result_dict['memory_usage_mb'] = float(line.split()[-1]) / 1e3
            if "Elapsed (wall clock) time" in line:
                result_dict['wall_time_sec'] = get_sec(line.split()[-1])
            if "User time" in line:
                result_dict['user_time_sec'] = float(line.split()[-1])
            if "System time" in line:
                result_dict['sys_time_sec'] = float(line.split()[-1])
            # MacOS performance metrics
            if "peak memory footprint" in line:
                result_dict['memory_usage_mb'] = float(line.split()[0]) / 1e6
            if "instructions retired" in line:
                result_dict['num_instructions'] = float(line.split()[0])
            if "real" in line:
                result_dict['wall_time_sec'] = float(line.split()[0])
                result_dict['user_time_sec'] = float(line.split()[2])
                result_dict['sys_time_sec'] = float(line.split()[4])
    except:
        if logger is None:
            print("Evalplus extraction failed: %s" % result_file)
            traceback.print_exc()
        else:
            logger.info("Evalplus extraction failed: %s" % result_file)
            logger.info(traceback.format_exc())
    finally:
        return result_dict


def get_sec(time_str):
    """Get seconds from time."""
    fields = [float(x) for x in time_str.split(':')]
    if len(fields) == 2:
        m, s = fields; h = 0.0
    else:
        h, m, s = fields
    return h * 3600.0 + m * 60.0 + s


def unzip(x):
    return [list(x) for x in zip(*x)]


def get_time(space=True, date=True):
    '''Creates a nicely formated timestamp'''
    if date:
        date_str = "%Y-%m-%d %H:%M:%S"
    else:
        date_str = "%H:%M:%S"

    if not space:
        date_str = date_str.replace(":", "-").replace(" ", "_")

    return datetime.datetime.now(timezone('US/Pacific')).strftime(date_str)


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
