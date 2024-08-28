#!/usr/bin/env python
import ast
import copy
from collections import defaultdict
import glob
import json
import operator
import os
import pprint
import random
import sys
import time
import traceback

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from scipy.stats import ttest_ind

from analysis_util import \
    get_fitness_file, load_checkpoint, get_checkpoints
from analysis_util import COLORS, LINEWIDTH, FIG_SIZE, PLOT_FMT, PROP_CYCLER
from role_ga import Individual, DEFAULT_MAIN_ROLE
from util import extract_evalplus, datetime_to_epoch
from util import get_indv_config, get_eval_config

# Directory to get results from
EXPERIMENT_DIRS = []
# Filter the top K experiments
FILTER_TOP_EXPERIMENTS = 8
# Blacklist for words in experiment names
EXPERIMENT_NAME_BLACKLIST = []
# Combine up to K (take avg and std) of experiments with same label
COMBINE_LABELS = None
# Filter out incomplete experiments
FILTER_INCOMPLETE_EXP = False

# Choose from fitness or true fitness
FITNESS_METRIC = 'fitness'
# Whether to plot wall time instead of generations
PLOT_WALL_TIME = False
# Whether to plot the mean fitness or not
PLOT_MEAN = True
# Maximum number of generations to plot
MAX_GEN = None
# Override intervals for plotting purposes
OVERRIDE_INTERVAL = None
# Min-max fitness range to plot in
FITNESS_RANGE = None
# Custom label for Y-axis/X-axis
Y_LABEL = None; X_LABEL = None
# Name of results file
OUT_FILE = ('%s_GEN-%s_RANGE-%s_METRIC-%s_comp.%s' % \
    ("-".join([os.path.basename(y) for x, y in EXPERIMENT_DIRS]),
        MAX_GEN, FITNESS_RANGE, FITNESS_METRIC, PLOT_FMT))[:255]


def _load_fitness(experiment_dir, max_gen=MAX_GEN, fit_metric=FITNESS_METRIC):
    if fit_metric == "fitness":
            fitness_file = get_fitness_file(experiment_dir, 'fitness.txt')
    else:
        assert fit_metric == "true_fitness"
        fitness_file = get_fitness_file(experiment_dir, 'true_fitness.txt')

    gen_dict = {}
    with open(fitness_file) as f:
        for line in f:
            if line.startswith("#"): continue
            try: d, t, g, b, m, s = line.rstrip().split()
            except: d, t, g, b, m = line.rstrip().split()
            t = datetime_to_epoch("%s %s" % (d, t))
            # t, g, b, m  = line.rstrip().split(); t = t.split("/")[0]
            gen_dict[int(g)] = (t, b, m)

    sorted_fit_list = sorted(gen_dict.items(), key=lambda x: x[0])
    r = [x[1] for x in sorted_fit_list]
    if max_gen is not None and len(r) > max_gen:
        r = r[:max_gen]

    time = [int(x[0]) for x in r]
    time = [x - min(time) for x in time]
    result_dict = {'time': time,
        'best': [float(x[1]) for x in r],
        'mean': [float(x[2]) for x in r],
        'gen': [x[0] for x in sorted_fit_list]}

    if len(result_dict.get('best')) == 0:
        return None
    return result_dict


def _get_experiment_dirs():
    expanded_dirs = []
    for experiment_label, experiment_dir in EXPERIMENT_DIRS:
        experiment_dirs = sorted([x for x in glob.glob(experiment_dir) \
            if not any([y in os.path.basename(x)
                for y in EXPERIMENT_NAME_BLACKLIST])])
        experiment_dirs = [x for x in experiment_dirs if os.path.isdir(x)]

        if FILTER_INCOMPLETE_EXP:
            _experiment_dirs = []
            for experiment_dir in experiment_dirs:
                if os.path.exists(os.path.join(experiment_dir, 'done')):
                    _experiment_dirs.append(experiment_dir)
            experiment_dirs = _experiment_dirs

        experiment_labels = [experiment_label] * len(experiment_dirs)
        expanded_dirs.extend(list(zip(experiment_labels, experiment_dirs)))
    return sorted(expanded_dirs)


# def combine_labels(experiment_dict):
#     combined_dict = defaultdict(list)

#     for (experiment_name, experiment_label), result_dict \
#         in experiment_dict.items():

#         combined_dict[experiment_label].append(result_dict)

#     new_experiment_dict = {}
#     for experiment_label, result_dicts in combined_dict.items():
#         if len(result_dicts) > COMBINE_LABELS:
#             result_dicts = random.sample(result_dicts, COMBINE_LABELS)

#         best_values = [rs.get('best') for rs in result_dicts]
#         print("Combined label: %s result lengths: %s" % (experiment_label,
#             [len(bv) for bv in best_values]))
#         max_length = max([len(bv) for bv in best_values])
#         for bv in best_values:
#             if len(bv) < max_length:
#                 bv.extend([np.nan]*(max_length - len(bv)))

#         combined_result_dict = {}
#         combined_result_dict['best'] = np.nanmean(best_values, axis=0)
#         combined_result_dict['best_std'] = np.nanstd(best_values, axis=0)
#         combined_result_dict['num_trials'] = len(result_dicts)

#         intervals = np.array([rs.get('interval') for rs in result_dicts])
#         assert np.all(intervals == intervals[0])
#         combined_result_dict['interval'] = intervals[0]

#         if OVERRIDE_INTERVAL is not None:
#             assert combined_result_dict['interval'] % OVERRIDE_INTERVAL == 0 \
#                 or OVERRIDE_INTERVAL % combined_result_dict['interval'] == 0
#             ratio = int(OVERRIDE_INTERVAL/combined_result_dict['interval'])
#             if ratio > 1:
#                 combined_result_dict['best'] = \
#                     combined_result_dict['best'][::ratio]
#                 combined_result_dict['best_std'] = \
#                     combined_result_dict['best_std'][::ratio]
#             elif ratio == 0:
#                 ratio = int(combined_result_dict['interval']/OVERRIDE_INTERVAL)
#                 combined_result_dict['interval'] = OVERRIDE_INTERVAL
#                 combined_result_dict['best'] = \
#                     np.repeat(combined_result_dict['best'], ratio)
#                 combined_result_dict['best_std'] = \
#                     np.repeat(combined_result_dict['best_std'], ratio)

#         new_experiment_dict[(experiment_label, experiment_label)] = \
#             combined_result_dict
#     return new_experiment_dict


# def t_test(experiment_dict):
#     def print_stats(label, result_dict):
#         if 'best_std' not in result_dict or 'best' not in result_dict:
#             return
#         std = result_dict.get("best_std")[-1] * 100
#         best = result_dict.get("best")[-1] * 100
#         print("%s: %.2f (%.2f)" % (label, best, std))

#     baselines = []
#     experiments = []
#     for (experiment_name, experiment_label), result_dict in \
#         experiment_dict.items():
#         if 'Baseline' in experiment_label:
#             baselines.append((experiment_label, result_dict))
#         else:
#             experiments.append((experiment_label, result_dict))

#     for exp_label, result_dict in experiments:
#         print_stats(exp_label, result_dict)
#     for b_label, b_result_dict in baselines:
#         print_stats(b_label, b_result_dict)

#     for exp_label, result_dict in experiments:
#         if 'best_std' not in result_dict or 'best' not in result_dict:
#             continue

#         best_baselines = [(l, max(rd.get('best'))) for l, rd in baselines]
#         exceeded = set()
#         total_epochs = len(result_dict.get('best')) * \
#             result_dict.get('interval')
#         for i, value in enumerate(result_dict.get('best')):
#             epochs = (i + 1) * result_dict.get('interval')
#             for l, bb in best_baselines:
#                 if value >= bb and l not in exceeded:
#                     print("Training percentage to exceed baseline %s: %s/%s" % \
#                         (l, epochs, total_epochs))
#                     exceeded.add(l)

#         for b_label, b_result_dict in baselines:

#             s, p = ttest_ind_from_stats(
#                 result_dict.get('best')[-1],
#                 result_dict.get('best_std')[-1],
#                 result_dict.get('num_trials'),
#                 b_result_dict.get('best')[-1],
#                 b_result_dict.get('best_std')[-1],
#                 b_result_dict.get('num_trials'),
#                 equal_var=False)

#             print("T-test %s/%s: %.2f" % (exp_label, b_label, p))


def _compare_experiments():
    experiment_dict = {}
    for experiment_label, experiment_dir in _get_experiment_dirs():
        experiment_name = os.path.basename(experiment_dir)
        try:
            result_dict = _load_fitness(experiment_dir, max_gen=MAX_GEN,
                fit_metric=FITNESS_METRIC)
        except:
            traceback.print_exc(); result_dict = None
        if result_dict is None: continue
        if experiment_label is None: experiment_label = experiment_name

        experiment_dict[(experiment_name, experiment_label)] = result_dict

    # if COMBINE_LABELS is not None:
    #     assert not PLOT_WALL_TIME
    #     experiment_dict = combine_labels(experiment_dict)

    if FILTER_TOP_EXPERIMENTS is not None:
        experiment_dict = dict(sorted(experiment_dict.items(), reverse=True,
            key=lambda x: max(x[1].get('best')))[:FILTER_TOP_EXPERIMENTS])

    assert len(experiment_dict) > 0
    # t_test(experiment_dict)

    plt.figure(figsize=FIG_SIZE)
    for i, ((experiment_name, experiment_label),
        result_dict) in enumerate(sorted(experiment_dict.items())):

        print("Plotting experiment fitnesses for %s" % experiment_name)
        if PLOT_WALL_TIME:
            if 'time' not in result_dict:
                continue
            epochs = result_dict.get('time')
        else:
            epochs = np.array(range(len(result_dict.get('best'))))

        kwargs = PROP_CYCLER[i]
        plt.plot(epochs, result_dict.get('best'),
            label='%s' % experiment_label, linewidth=LINEWIDTH, **kwargs)
        if 'best_std' in result_dict:
            yerr=np.array(result_dict.get('best_std'))/ \
                math.sqrt(result_dict.get('num_trials'))
            plt.fill_between(epochs,
                result_dict.get('best') - yerr,
                result_dict.get('best') + yerr,
                alpha=0.2, facecolor=kwargs.get('color'),
                antialiased=True)

        if PLOT_MEAN and 'mean' in result_dict:
            plt.plot(epochs, result_dict.get('mean'),
                label='mean %s' % experiment_label, alpha=0.5,
                linewidth=LINEWIDTH * 0.75, **kwargs)

    if FITNESS_RANGE is not None:
        plt.ylim(FITNESS_RANGE)

    handles, labels = plt.gca().get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)
    plt.legend(handles2, labels2)
    plt.grid(which='major', axis='x')
    plt.grid(which='major', axis='y')
    plt.grid(which='minor', linestyle='--', axis='y')

    if X_LABEL is not None: plt.xlabel(X_LABEL)
    elif PLOT_WALL_TIME: plt.xlabel("Wall Time (Seconds)")
    else: plt.xlabel("Generations")

    default_y_label = FITNESS_METRIC.title().replace("_", " ")
    plt.ylabel(Y_LABEL) if Y_LABEL is not None else plt.ylabel(default_y_label)
    plt.savefig(OUT_FILE, bbox_inches='tight', dpi=200)


def compare_experiments_main():
    _experiment_dirs = [
        [('Single Agent', 'results/8_20_multirole_coding_prompt'),
        ('Multi Agent', 'results/8_19_multirole_coding_prompt')],

        [('Single Agent', 'results/8_20_multirole_coding_prompt'),
        ('Multi Agent (8/19)', 'results/8_19_multirole_coding_prompt'),
        ('Multi Agent (8/17)', 'results/8_17_multirole'),
        ('Multi Agent (8/6)', 'results/8_6_multirole')],
    ]
    _blacklists = [[]] * len(_experiment_dirs)
    _combine_labels = [None] * len(_experiment_dirs)
    _fitness_metrics = ['fitness', 'true_fitness']

    global EXPERIMENT_DIRS, EXPERIMENT_NAME_BLACKLIST
    global COMBINE_LABELS, FITNESS_METRIC, OUT_FILE
    for x in zip(_experiment_dirs, _blacklists, _combine_labels, _fitness_metrics):
        print(x); EXPERIMENT_DIRS, EXPERIMENT_NAME_BLACKLIST, COMBINE_LABELS, \
            FITNESS_METRIC = x
        exp_names = '-'.join([os.path.basename(y) for x, y in EXPERIMENT_DIRS])
        OUT_FILE = ('results/%s_MAXGEN-%s_RANGE-%s_METRIC-%s.%s' % \
            (exp_names, MAX_GEN, FITNESS_RANGE, FITNESS_METRIC, PLOT_FMT))[:255]
        _compare_experiments()


def multirun_evalplus(main_prompt=DEFAULT_MAIN_ROLE,
    team_prompt='config/autogen_builder_init2.json',
    evolve_mode='team',
    indv=None,
    exp_name=None,
    use_prompt=True,
    n_trials=10,
    n_workers=10,
    dataset='humaneval',
    eval_config={},
    result_dir=None,
    baseline_result_dir="results/multirole_baseline",
    seed=0):

    random.seed(seed); np.random.seed(seed)
    if result_dir is None: result_dir = "."

    results_file = os.path.join(result_dir, "evalplus_results.yaml")
    eval_results_list = glob.glob(os.path.join(result_dir, "*/evalplus.txt"))
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            evalplus_results = YAML().load(f)
    elif len(eval_results_list) == n_trials:
        evalplus_results = []
        for evalplus_fp in eval_results_list:
            evalplus_results.append(extract_evalplus(evalplus_fp))
    else:
        from llm_evaluator import LLMEvaluator; assert n_trials > 0
        if not use_prompt: assert indv is not None; _id = indv.id; t = None
        else: _id = None; t = int(time.time())

        result_dir = "results/multirun_%s_%s_N-%s_T-%s" % \
            (exp_name, _id, n_trials, t)
        os.makedirs(result_dir, exist_ok=True)

        if use_prompt:
            if os.path.exists(main_prompt):
                with open(main_prompt, "r") as f: main_prompt = f.read()
            if os.path.exists(team_prompt):
                with open(team_prompt, "r") as f: team_prompt = json.load(f)

            population = [Individual({}) for i in range(n_trials)]
            for indv in population:
                indv.main_role = main_prompt; indv.team_role = team_prompt
                indv.evolve_mode = evolve_mode
            print("Creating individual from main/team prompts")
            print(population[0])
        else:
            population = [indv.create_child(indv.gen_created) \
                for i in range(n_trials)]

        with open(os.path.join(result_dir, 'indv.yaml'), 'w') as f:
            YAML().dump(population[0].serialize(), f)
        with open(os.path.join(result_dir, 'indv_config.yaml'), 'w') as f:
            YAML().dump(population[0].config, f)

        eval_config['n_workers'] = n_workers
        eval_config['dataset'] = dataset
        evaluator = LLMEvaluator(eval_config, evaluator_dir=result_dir)
        result_dicts = evaluator.evaluate(population)
        evalplus_results = [x.get('evalplus_result', {}) for x in result_dicts]

    baseline_results = None
    if baseline_result_dir is not None:
        base_results_file = os.path.join(baseline_result_dir,
            'evalplus_results.yaml'); assert os.path.exists(base_results_file)
        with open(base_results_file, "r") as f:
            baseline_results = YAML().load(f)
        assert len(baseline_results) > 0

    print("Computing statistics from evalplus results: %s" % result_dir)
    assert len(evalplus_results) > 0
    with open(os.path.join(result_dir, 'summary.txt'), 'w') as f:
        for key in evalplus_results[0]:
            _results = [x[key] for x in evalplus_results]
            mean = np.mean(_results); std = np.std(_results)
            f.write("%s mean (std): %.3f (%.3f)\n" % (key, mean, std))

        if baseline_results is not None:
            print("Computing t-tests from baseline results: %s" % \
                baseline_result_dir); f.write("\n\n")
            f.write("Baseline: %s\n" % baseline_result_dir)
            for key in evalplus_results[0]:
                if key not in baseline_results[0]: continue
                _baselines = [x[key] for x in baseline_results]
                _results = [x[key] for x in evalplus_results]
                ttest_result = ttest_ind(_baselines, _results, equal_var=False)
                mean = np.mean(_baselines); std = np.std(_baselines)
                f.write("%s p-value (baseline: %.3f (%.3f)): %.3f\n" % \
                    (key, mean, std, ttest_result.pvalue))

    with open(os.path.join(result_dir, 'evalplus_results.yaml'), 'w') as f:
        YAML().dump(evalplus_results, f)


def multirun_evalplus_exp(experiment_dir,
    top_n=1,
    min_samples=3,
    agg_func=np.mean, # np.mean, np.median, np.max, lambda x: x[-1]
    gen_range=(0, 999),
    use_true_fitness=False,
    eval_indv=False,
    *args,
    **kwargs):

    indv_config = get_indv_config(experiment_dir)
    eval_config = get_eval_config(experiment_dir)
    _fit_list_dict = defaultdict(list)
    _true_fit_list_dict = defaultdict(list)
    _indv_dict = {}
    checkpoints = get_checkpoints(experiment_dir, min_gen=gen_range[0],
        max_gen=gen_range[1])
    for checkpoint in checkpoints:
        pop_dict = load_checkpoint(checkpoint)
        for indv_dict in pop_dict:
            indv = Individual(config=indv_config); indv.deserialize(indv_dict)
            if use_true_fitness: fitness = indv.get_true_fitness()
            else: fitness = indv.get_fitness(raw_fitness=True)

            _fit_list_dict[indv.id].append(fitness)
            _true_fit_list_dict[indv.id].append(indv.get_true_fitness())
            _indv_dict[indv.id] = indv

    for indv_id, fit_list in _fit_list_dict.items():
        if len(fit_list) < min_samples:
            del _indv_dict[indv_id]; continue
        agg_fit = agg_func(fit_list)
        true_agg_fit = agg_func(_true_fit_list_dict[indv_id])
        _indv_dict[indv_id].set_fitness(agg_fit)
        _indv_dict[indv_id].set_true_fitness(true_agg_fit)

    best_indv = sorted(_indv_dict.values(), reverse=True)[:top_n]

    for i, indv in enumerate(best_indv):
        print("#### agg_func: %s, rank: %s, samples: %s ####" % \
            (agg_func.__name__, i + 1, len(_fit_list_dict[indv.id])))
        print(indv); print("\n\n")
        if eval_indv:
            multirun_evalplus(indv=indv,
                exp_name=os.path.basename(experiment_dir.rstrip("/")),
                use_prompt=False,
                eval_config=eval_config,
                *args,
                **kwargs)


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


def compare_agent_chat_stats(experiment_dir,
    top_n=5,
    agg_func=np.mean,
    gen_range=(10, None),
    indv_quartile=[0.8, 1.0]):

    print("Experiment dir: %s" % experiment_dir)
    indv_list = []; indv_config = get_indv_config(experiment_dir)
    checkpoints = get_checkpoints(experiment_dir, min_gen=gen_range[0],
        max_gen=gen_range[1])
    for checkpoint in checkpoints:
        pop_dict = load_checkpoint(checkpoint)
        for indv_dict in pop_dict:
            indv = Individual(config=indv_config); indv.deserialize(indv_dict)
            indv_list.append(indv)

    indv_list = sorted(indv_list); n = len(indv_list)
    a, b = indv_quartile; assert a >= 0 and b >= 0 and a <= 1.0 and b <= 1.0
    indv_list = indv_list[int(a * n):int(b * n)]; assert len(indv_list) > 0
    print("Getting chat stats for %s indv (avg fit: %.4f) in quartile %s from gen %s" % \
        (len(indv_list), np.mean([x.get_fitness(True) for x in indv_list]),
            indv_quartile, gen_range))

    _chat_count_dict = defaultdict(list); _code_count_dict = defaultdict(list)
    for indv in indv_list:
        for agent, chat_count in indv.eval_stats['agent_chat_count'].items():
            _chat_count_dict[agent].append(chat_count)
        for agent, code_count in indv.eval_stats['agent_code_count'].items():
            _code_count_dict[agent].append(code_count)

    chat_count_dict = {}; code_count_dict = {}
    for k, v in _chat_count_dict.items(): chat_count_dict[k] = agg_func(v)
    for k, v in _code_count_dict.items(): code_count_dict[k] = agg_func(v)

    best_chat_count = sorted(chat_count_dict.items(), key=lambda x: x[1],
        reverse=True)[:top_n]
    best_code_count = sorted(code_count_dict.items(), key=lambda x: x[1],
        reverse=True)[:top_n]
    worst_chat_count = sorted(chat_count_dict.items(), key=lambda x: x[1],
        reverse=False)[:top_n]
    worst_code_count = sorted(code_count_dict.items(), key=lambda x: x[1],
        reverse=False)[:top_n]

    print("Top chat count agents:\n%s" % best_chat_count)
    print("Bottom chat count agents:\n%s" % worst_chat_count)
    print("Top code count agents:\n%s" % best_code_count)
    print("Bottom code count agents:\n%s" % worst_code_count)
    print("\n")

if __name__ == "__main__":
    multirun_evalplus()
    # multirun_evalplus_exp(sys.argv[1], use_true_fitness=True, eval_indv=True)
    # compare_experiments_main()
    # generate_evalplus_weights_file(sys.argv[1])
    # compare_agent_chat_stats(sys.argv[1], indv_quartile=[0.0, 1.0])
