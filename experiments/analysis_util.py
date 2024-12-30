import functools
import glob
import os
import pickle

from cycler import cycler
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np
from ruamel.yaml import YAML

# Colors for plotting
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k'] * 100
# Size of figure for plotting
FIG_SIZE = (12, 5)
# Different line styles for plotting
LINESTYLES = ['-', '--', '-.',':'] * 100
# Default line width for plotting
LINEWIDTH = 1.5
# Get true fitness from indv dict
TRUE_FITNESS = True
# Format for plots
PLOT_FMT = 'png'
# Minimum fitness for sorting purposes
MIN_FITNESS = 1e-12
# Cycler between styles for lots of lines
PROP_CYCLER = list((cycler(linestyle=LINESTYLES[:4]) * \
    cycler(color=COLORS[:7]))) * 100


def is_outlier(points, threshold=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An num-observations by num-dimensions array of observations
        threshold : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A num-observations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > threshold


def get_fitness(indv_dict, override_fitness=None):
    from role_ga import Individual
    if isinstance(indv_dict, Individual):
        indv_dict = indv_dict.serialize()
    if TRUE_FITNESS:
        key = 'true_fitness'
    else:
        key = 'fitness'
    if override_fitness is not None:
        key = override_fitness
    fitness = indv_dict.get(key)

    if fitness is None:
        fitness = MIN_FITNESS

    return fitness


def get_config(experiment_dir):
    config_file = os.path.join(experiment_dir, "config.yaml")
    assert os.path.exists(config_file), \
        "config file: %s does not exist" % config_file
    with open(config_file, 'r') as f:
        config = YAML().load(f)
    return config


def get_fitness_file(experiment_dir, override_fitness_file=None):
    fitness_file_name = "true_fitness.txt" if TRUE_FITNESS else "fitness.txt"
    if override_fitness_file is not None:
        fitness_file_name = override_fitness_file
    fp = os.path.join(experiment_dir, "role_ga/%s" % fitness_file_name)
    if not os.path.exists(fp):
        # Check directly in exp dir if role_ga does not exists
        fp = os.path.join(experiment_dir, fitness_file_name)
        assert os.path.exists(fp)
    return fp


def get_checkpoints(experiment_dir, is_base_dir=True, min_gen=None,
    max_gen=None, verbose=True):
    assert os.path.exists(experiment_dir)
    if is_base_dir:
        checkpoints = glob.glob(os.path.join(experiment_dir,
            "role_ga/*.yaml"))
    else:
        checkpoints = glob.glob(os.path.join(experiment_dir,
            "*.yaml"))
    checkpoints = sorted(checkpoints,
        key=lambda x: int(x.rstrip(".yaml").split("_")[-1]))

    if min_gen is None:
        min_gen = 0
    if max_gen is None:
        max_gen = len(checkpoints)

    checkpoints = sorted(checkpoints,
        key=lambda x: int(x.rstrip(".yaml").split("_")[-1]))
    within_range = checkpoints[min_gen:max_gen]

    # within_range = []
    # for checkpoint in checkpoints:
    #     n = int(checkpoint.rstrip(".yaml").split("_")[-1])
    #     if n >= min_gen and n < max_gen:
    #         within_range.append(checkpoint)
    if verbose:
        print("Retrieved %s checkpoints" % len(within_range))
    return within_range


@functools.lru_cache(maxsize=None, typed=False)
def load_checkpoint(checkpoint, gen=False, cache=True):
    cache_file = checkpoint + ".pkl"
    if cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return_data = pickle.load(f)
        except:
            os.remove(cache_file)
            return load_checkpoint(checkpoint, gen, cache)
    else:
        assert os.path.exists(checkpoint)
        with open(checkpoint, "r") as f:
            pop_list = YAML().load(f)
        sorted_pop_list = sorted(pop_list.get("individuals"), reverse=True,
            key=lambda x: get_fitness(x))
        return_data = (sorted_pop_list, pop_list.get("generation"))

    if cache and not os.path.exists(cache_file):
        with open(cache_file, 'wb') as f:
            pickle.dump(return_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    if not gen:
        return_data = return_data[0]
    print("Parsed following checkpoint: %s" % os.path.basename(checkpoint))
    return return_data


def generate_evalplus_weights_file(jsons_dir,
    result_dir=".",
    min_weight=0.0,
    max_weight=1.0,
    use_quantile_weights=True,
    quantiles_probs=[0.25, 0.5, 0.75, 1.0],
    quantile_weights=[0.25, 0.5, 0.75, 1.0]):

    def normalize(v):
        return v * (max_weight - min_weight) + min_weight

    def get_weights(score_count, total_count):
        for k, v in score_count.items():
            score_count[k] = normalize(v/total_count)

        if use_quantile_weights:
            assert len(quantiles_probs) == len(quantile_weights)
            quantiles = np.quantile(list(score_count.values()), quantiles_probs)
            # print("Quantiles: %s" % quantiles)
            weights_dict = {}
            for k, v in score_count.items():
                for q, qw in zip(quantiles, quantile_weights):
                    if q >= v:
                        weights_dict[k] = qw; break
            return weights_dict
        else:
            return score_count

    base_count = {}; plus_count = {}; _b = {}; _p = {}; total_count = 0.0
    for eval_json in glob.glob(os.path.join(jsons_dir, "**/eval_results.json"),
        recursive=True):

        print("Processing %s" % eval_json); total_count += 1.0
        with open(eval_json, 'r') as f: eval_dict = json.load(f)

        for task_id, result in eval_dict['eval'].items():
            if task_id not in base_count:
                base_count[task_id] = 0.0; _b[task_id] = 0.0
            if task_id not in plus_count:
                plus_count[task_id] = 0.0; _p[task_id] = 0.0

            if result[0]['base_status'] != "pass": base_count[task_id] += 1.0
            else: _b[task_id] += 1.0
            if result[0]['plus_status'] != "pass": plus_count[task_id] += 1.0
            else: _p[task_id] += 1.0

    for k in base_count:
        assert k in _b; assert _b[k] + base_count[k] == total_count
    for k in plus_count:
        assert k in _p; assert _p[k] + plus_count[k] == total_count
    print("Processed %d results" % total_count)

    base_weights = get_weights(base_count, total_count)
    plus_weights = get_weights(plus_count, total_count)
    bw = list(base_weights.values()); pw = list(plus_weights.values())
    weights_dict = {'base_weights': base_weights,
        'plus_weights': plus_weights,
        'base_weights_mean': np.mean(bw),
        'base_weights_std': np.std(bw),
        'plus_weights_mean': np.mean(pw),
        'plus_weights_std': np.std(pw)}

    outfile = os.path.join(result_dir,
        os.path.basename(jsons_dir) + "_weights.json")
    with open(outfile, 'w') as f: json.dump(weights_dict, f, indent=4)
    pprint.pprint(weights_dict)


# def collect_ancestry_chain(experiment_dir, indv_id, initial_gen):
#     experiment_dir = glob.glob(experiment_dir)[0] # cleanup wild cards
#     config = get_config(experiment_dir)
#     domain = config.get("domain_name")
#     epoch_per_gen = config.get(domain + "_config").get("num_epochs")
#     assert epoch_per_gen is not None

#     if initial_gen == "last":
#         curr_checkpoint = get_checkpoints(experiment_dir)[-1]
#         pop_list, initial_gen = load_checkpoint(curr_checkpoint, gen=True)
#     else:
#         curr_checkpoint = os.path.join(experiment_dir,
#             "role_ga/checkpoint_%s.yaml" % initial_gen)
#     if indv_id.startswith("rank_"):
#         if 'pop_list' not in locals():
#             pop_list, initial_gen = load_checkpoint(curr_checkpoint, gen=True)
#         rank = int(indv_id.split("_")[-1]) - 1
#         assert rank >= 0 and rank < len(pop_list)
#         indv_id = max(pop_list, key=lambda x: get_fitness(x)).get('id')
#         # indv_id = pop_list[rank].get("id")

#     print("Collecting ancestry for indv %s" % indv_id)
#     curr_indv_id = indv_id
#     curr_gen = initial_gen
#     ancestry_chain = {}
#     while True:
#         pop_list, gen = load_checkpoint(curr_checkpoint, gen=True)

#         found_next_indv = False
#         for indv in pop_list:
#             if indv.get("id") == curr_indv_id:
#                 if indv.get("gen_created") != curr_gen:
#                     curr_gen = indv.get("gen_created")
#                     found_next_indv = True
#                     break
#                 else:
#                     assert indv.get("gen_created") == gen
#                     training_info = indv.get("training_info")
#                     history = training_info.get("history")
#                     num_epochs_trained = training_info.get(
#                         "num_epochs_trained")
#                     num_epochs_at_start = num_epochs_trained - epoch_per_gen
#                     ancestry_chain[gen] = (num_epochs_at_start, indv)

#                     if len(indv.get("ancestry")) > 0:
#                         curr_gen = indv.get("ancestry")[0].get(
#                             "gen_created")
#                         curr_indv_id = indv.get("ancestry")[0].get("id")
#                         found_next_indv = True
#                     break

#         if not found_next_indv:
#             break
#         else:
#             curr_checkpoint = os.path.join(experiment_dir,
#                 "role_ga/checkpoint_%s.yaml" % curr_gen)

#     assert(len(ancestry_chain)) > 0
#     return ancestry_chain
