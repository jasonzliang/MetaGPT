import copy
import json
import os
import re
import sys
import random
import traceback
import time

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from analysis_util import \
    get_fitness_file, load_checkpoint, get_checkpoints
from analysis_util import COLORS, FIG_SIZE, PLOT_FMT, PROP_CYCLER
from llm_evaluator import LLMEvaluator

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


def load_fitness(experiment_dir, max_gen=MAX_GEN, fit_metric=FITNESS_METRIC):
    if fit_metric == "fitness":
            fitness_file = get_fitness_file(experiment_dir, 'fitness.txt')
    else:
        assert fit_metric == "true_fitness"
        fitness_file = get_fitness_file(experiment_dir, 'true_fitness.txt')

    gen_dict = {}
    with open(fitness_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            t, g, b, m = line.rstrip().split()
            t = t.split("/")[0]
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


def get_experiment_dirs():
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


def combine_labels(experiment_dict):
    combined_dict = defaultdict(list)

    for (experiment_name, experiment_label), result_dict \
        in experiment_dict.items():

        combined_dict[experiment_label].append(result_dict)

    new_experiment_dict = {}
    for experiment_label, result_dicts in combined_dict.items():
        if len(result_dicts) > COMBINE_LABELS:
            result_dicts = random.sample(result_dicts, COMBINE_LABELS)

        best_values = [rs.get('best') for rs in result_dicts]
        print("Combined label: %s result lengths: %s" % (experiment_label,
            [len(bv) for bv in best_values]))
        max_length = max([len(bv) for bv in best_values])
        for bv in best_values:
            if len(bv) < max_length:
                bv.extend([np.nan]*(max_length - len(bv)))

        combined_result_dict = {}
        combined_result_dict['best'] = np.nanmean(best_values, axis=0)
        combined_result_dict['best_std'] = np.nanstd(best_values, axis=0)
        combined_result_dict['num_trials'] = len(result_dicts)

        intervals = np.array([rs.get('interval') for rs in result_dicts])
        assert np.all(intervals == intervals[0])
        combined_result_dict['interval'] = intervals[0]

        if OVERRIDE_INTERVAL is not None:
            assert combined_result_dict['interval'] % OVERRIDE_INTERVAL == 0 \
                or OVERRIDE_INTERVAL % combined_result_dict['interval'] == 0
            ratio = int(OVERRIDE_INTERVAL/combined_result_dict['interval'])
            if ratio > 1:
                combined_result_dict['best'] = \
                    combined_result_dict['best'][::ratio]
                combined_result_dict['best_std'] = \
                    combined_result_dict['best_std'][::ratio]
            elif ratio == 0:
                ratio = int(combined_result_dict['interval']/OVERRIDE_INTERVAL)
                combined_result_dict['interval'] = OVERRIDE_INTERVAL
                combined_result_dict['best'] = \
                    np.repeat(combined_result_dict['best'], ratio)
                combined_result_dict['best_std'] = \
                    np.repeat(combined_result_dict['best_std'], ratio)

        new_experiment_dict[(experiment_label, experiment_label)] = \
            combined_result_dict
    return new_experiment_dict


def t_test(experiment_dict):
    def print_stats(label, result_dict):
        if 'best_std' not in result_dict or 'best' not in result_dict:
            return
        std = result_dict.get("best_std")[-1] * 100
        best = result_dict.get("best")[-1] * 100
        print("%s: %.2f (%.2f)" % (label, best, std))

    baselines = []
    experiments = []
    for (experiment_name, experiment_label), result_dict in \
        experiment_dict.items():
        if 'Baseline' in experiment_label:
            baselines.append((experiment_label, result_dict))
        else:
            experiments.append((experiment_label, result_dict))

    for exp_label, result_dict in experiments:
        print_stats(exp_label, result_dict)
    for b_label, b_result_dict in baselines:
        print_stats(b_label, b_result_dict)

    for exp_label, result_dict in experiments:
        if 'best_std' not in result_dict or 'best' not in result_dict:
            continue

        best_baselines = [(l, max(rd.get('best'))) for l, rd in baselines]
        exceeded = set()
        total_epochs = len(result_dict.get('best')) * \
            result_dict.get('interval')
        for i, value in enumerate(result_dict.get('best')):
            epochs = (i + 1) * result_dict.get('interval')
            for l, bb in best_baselines:
                if value >= bb and l not in exceeded:
                    print("Training percentage to exceed baseline %s: %s/%s" % \
                        (l, epochs, total_epochs))
                    exceeded.add(l)

        for b_label, b_result_dict in baselines:

            s, p = ttest_ind_from_stats(
                result_dict.get('best')[-1],
                result_dict.get('best_std')[-1],
                result_dict.get('num_trials'),
                b_result_dict.get('best')[-1],
                b_result_dict.get('best_std')[-1],
                b_result_dict.get('num_trials'),
                equal_var=False)

            print("T-test %s/%s: %.2f" % (exp_label, b_label, p))


def compare_experiments():
    experiment_dict = {}
    for experiment_label, experiment_dir in get_experiment_dirs():
        experiment_name = os.path.basename(experiment_dir)
        try:
            result_dict = load_fitness(experiment_dir)
        except:
            result_dict = None

        if result_dict is None:
            continue
        if experiment_label is None:
            experiment_label = experiment_name
        experiment_dict[(experiment_name, experiment_label)] = result_dict

    if COMBINE_LABELS is not None:
        assert not PLOT_WALL_TIME
        experiment_dict = combine_labels(experiment_dict)

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
            epochs = result_dict.get('interval') * \
                (np.array((range(len(result_dict.get('best'))))) + 1)

        kwargs = PROP_CYCLER[i]
        plt.plot(epochs, result_dict.get('best'),
            label='%s' % experiment_label, **kwargs)
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
                label='mean %s' % experiment_label, alpha=0.5, **kwargs)

    if FITNESS_RANGE is not None:
        plt.ylim(FITNESS_RANGE)

    handles, labels = plt.gca().get_legend_handles_labels()
    hl = sorted(zip(handles, labels), key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)
    plt.legend(handles2, labels2)

    plt.grid(b=True, which='major', axis='x')
    plt.grid(b=True, which='major', axis='y')
    plt.grid(b=True, which='minor', linestyle='--', axis='y')

    if X_LABEL is not None:
        plt.xlabel(X_LABEL)
    elif PLOT_WALL_TIME:
        plt.xlabel("Wall Time (Seconds)")
    else:
        plt.xlabel("Epochs of Training")
    plt.ylabel(Y_LABEL) if Y_LABEL is not None else plt.ylabel('Fitness')
    plt.savefig(OUT_FILE, bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    _experiment_dirs = [
        [('5/19 Role GA', '../results/5_19_role_ga')],
    ]

    _blacklists = [[]] * len(_experiment_dirs)
    _combine_labels = [None] * len(_experiment_dirs)

    for x in zip(_experiment_dirs, _blacklists, _combine_labels):
        print(x)
        EXPERIMENT_DIRS, EXPERIMENT_NAME_BLACKLIST, COMBINE_LABELS = x
        labels = EXPERIMENT_DIRS; assert len(labels) > 0

        OUT_FILE = ('%s_TIME-%s_GEN-%s_RAN-%s_MET-%s_comp.%s' % \
            ("-".join([os.path.basename(y) for x, y in labels]),
            int(time.time()), MAX_GEN, FITNESS_RANGE, FITNESS_METRIC,
            PLOT_FMT))[:255]
        compare_experiments()
