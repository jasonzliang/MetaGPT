import copy
import datetime
import os
import pickle
import sys
import traceback

import jsonpickle
import numpy as np
from pytz import timezone
import ruamel.yaml as yaml

from alg_util import is_numpy_type, randomword

def extract_evalplus_score(result_file, logger=None):
    try:
        with open(result_file, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if "humaneval (base tests)" in line:
                scoreline = lines[i+1]
        score = float(scoreline.rstrip().rsplit()[1])
        assert 0.0 <= score <= 1.0
        return score
    except:
        if logger is None:
            print("Evalplus score extraction failed: %s" % result_file)
        else:
            logger.debug("Evalplus score extraction failed: %s" % result_file)
        return 0.0

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
        yaml.dump(config_dict, f)
    print("Dumped global config to file: %s" % filepath)