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

from adjustText import adjust_text
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import YAML
from scipy.stats import ttest_ind
import seaborn as sns

from analysis_util import get_fitness_file, load_checkpoint, get_checkpoints
from analysis_util import generate_evalplus_weights_file
from analysis_util import COLORS, LINEWIDTH, FIG_SIZE, PLOT_FMT, PROP_CYCLER
from role_ga import Individual
from llm_operators import DEFAULT_MAIN_ROLE, DEFAULT_MAIN_ROLE_V2
from self_improve import _load_checkpoint as load_self_improve_chkpt
from util import extract_evalplus, datetime_to_epoch, glob_result_dirs
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


def _compare_evo_experiments():
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

    if FILTER_TOP_EXPERIMENTS is not None:
        experiment_dict = dict(sorted(experiment_dict.items(), reverse=True,
            key=lambda x: max(x[1].get('best')))[:FILTER_TOP_EXPERIMENTS])

    assert len(experiment_dict) > 0; plt.figure(figsize=FIG_SIZE)
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
                alpha=0.5, linewidth=LINEWIDTH * 0.75, **kwargs)

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


def compare_evo_experiments_main():
    _experiment_dirs = [
        [('Single-agent + weighted fitness/coding instruct (8/20)', 'results/8_20_multirole_coding_prompt'),
        ('Multi-agent + weighted fitness/coding instruct (8/19)', 'results/8_19_multirole_coding_prompt')],

        [('Single-agent + weighted fitness/coding instruct (8/20)', 'results/8_20_multirole_coding_prompt'),
        ('Multi-agent + weighted fitness/coding instruct (8/19)', 'results/8_19_multirole_coding_prompt'),
        ('Multi-agent + weighted fitness (8/17)', 'results/8_17_multirole'),
        ('Multi-agent (8/6)', 'results/8_6_multirole')],

        [('8/30', 'results/8_30_multirole_coding_prompt'),
        ('8/31', 'results/8_31_multirole_coding_prompt')],
    ]
    _blacklists = [[]] * len(_experiment_dirs)
    _combine_labels = [None] * len(_experiment_dirs)
    _fitness_metrics = ['true_fitness', 'true_fitness', 'true_fitness']

    global EXPERIMENT_DIRS, EXPERIMENT_NAME_BLACKLIST
    global COMBINE_LABELS, FITNESS_METRIC, OUT_FILE
    for x in zip(_experiment_dirs, _blacklists, _combine_labels, _fitness_metrics):
        print(x); EXPERIMENT_DIRS, EXPERIMENT_NAME_BLACKLIST, COMBINE_LABELS, \
            FITNESS_METRIC = x
        exp_names = '-'.join([os.path.basename(y) for x, y in EXPERIMENT_DIRS])
        OUT_FILE = ('results/%s_MAXGEN-%s_RANGE-%s_METRIC-%s.%s' % \
            (exp_names, MAX_GEN, FITNESS_RANGE, FITNESS_METRIC, PLOT_FMT))[:255]
        _compare_evo_experiments()


def multirun_evalplus(
    use_prompt=True,
    main_prompt=DEFAULT_MAIN_ROLE_V2,
    team_prompt='config/8_3_best_multirole.json',
    evolve_mode='team',
    indv=None,
    experiment_dir='config',
    config_name='role_evo_multirole.yaml',
    n_trials=10,
    n_workers=10,
    dataset='humaneval',
    result_dir='results/multirun_config_None_N-10_T-1729576605',
    baseline_result_dir=None,
    seed=0):

    random.seed(seed); np.random.seed(seed); assert n_trials > 0
    if experiment_dir is not None:
        assert os.path.exists(experiment_dir)
        exp_name = os.path.basename(experiment_dir.rstrip("/"))
    else: exp_name = None
    if not use_prompt: assert indv is not None; _id = indv.id; t = None
    else: _id = None; t = int(time.time())
    if result_dir is None:
        result_dir = "results/multirun_%s_%s_N-%s_T-%s" % \
            (exp_name, _id, n_trials, t)
    os.makedirs(result_dir, exist_ok=True)

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
        from llm_evaluator import EvalPlusEvaluator
        if use_prompt:
            if experiment_dir is not None:
                indv_config = get_indv_config(experiment_dir,
                    config_name=config_name)
            else: indv_config = {}

            if os.path.exists(main_prompt):
                with open(main_prompt, "r") as f: main_prompt = f.read()
            if os.path.exists(team_prompt):
                with open(team_prompt, "r") as f: team_prompt = json.load(f)

            population = [Individual(indv_config) for i in range(n_trials)]
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

        if experiment_dir is not None:
            eval_config = get_eval_config(experiment_dir, config_name)
        else: eval_config = {}
        eval_config['n_workers'] = n_workers; eval_config['dataset'] = dataset
        print("Running %s trials with evaluator" % n_trials); time.sleep(3)

        evaluator = EvalPlusEvaluator(eval_config, evaluator_dir=result_dir)
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
            f.write("Comparison to baseline: %s\n" % baseline_result_dir)
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
    return result_dir


def multirun_evalplus_exp(
    experiment_dir,
    top_n=1,
    min_samples=3,
    agg_func=np.median, # np.mean, np.median, np.max, lambda x: x[-1]
    gen_range=(0, 999),
    use_true_fitness=False,
    eval_indv=False,
    *args,
    **kwargs):

    indv_config = get_indv_config(experiment_dir)
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

    best = sorted(_indv_dict.values(), reverse=True)[:top_n]; result_dirs = []
    for i, indv in enumerate(best):
        print("#### agg_func: %s, rank: %s, samples: %s ####" % \
            (agg_func.__name__, i + 1, len(_fit_list_dict[indv.id])))
        print(indv); print("\n\n")
        if not eval_indv: continue
        result_dir = multirun_evalplus(indv=indv,
            experiment_dir=experiment_dir,
            use_prompt=False,
            *args,
            **kwargs)
        result_dirs.append(result_dir)
    return result_dirs


def compare_agent_chat_stats(
    experiment_dir,
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


def visualize_self_improve_perf(
    result_dirs,
    use_glob=True,
    key='num_tries',
    key_filter=('gen_solved', None, 0),
    out_dir='results/'):

    if use_glob: result_dirs = glob_result_dirs(result_dirs)

    solution_dict = defaultdict(list); solved_counter = defaultdict(list)
    for result_dir in result_dirs:
        checkpoint_dict = load_self_improve_chkpt(result_dir)
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


def compare_scicode_evals(
    result_dirs,
    use_glob=True,
    with_background=False,
    test_set=True,
    label_top_k=8,
    out_dir='results/'):

    def _scatter_label(ax, n, x, y):
        texts = []
        for i in range(n):
            texts.append(ax.text(x[i], y[i], i+1,
                horizontalalignment='left', size='large', color='black',
                weight='semibold'))
        adjust_text(texts,
           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
           expand=(6, 6))

    if use_glob: result_dirs = glob_result_dirs(result_dirs)
    if with_background: result_name = 'scicode_eval_with_background.txt'
    else: result_name = 'scicode_eval_without_background.txt'
    if test_set: gd_total = (288, 65)
    else: gd_total = (50, 15)

    result_list = []
    for result_dir in result_dirs:
        result_file = glob.glob(os.path.join(result_dir, result_name))
        if len(result_file) == 0: continue
        else: result_file = result_file[0]

        result_dict = {'name': os.path.basename(result_dir),
            'time': os.path.getmtime(result_file)}

        with open(result_file, 'r') as f:
            for line in f.readlines():
                if line.lower().startswith("correct"):
                    try:
                        value = int(line.split(':')[1].strip().split('/')[0])
                        total = int(line.split(':')[1].strip().split('/')[1])
                    except:
                        continue

                    if "steps" in line:
                        result_dict['solved_sub_problems'] = value
                        result_dict['sub_problems_total'] = total
                    elif "problems" in line:
                        result_dict['solved_problems'] = value
                        result_dict['problems_total'] = total

        try:
            assert 'solved_problems' in result_dict
            assert 'solved_sub_problems' in result_dict
            assert result_dict['sub_problems_total'] == gd_total[0]
            assert result_dict['problems_total'] == gd_total[1]
        except:
            continue

        pprint.pprint(result_dict)
        result_list.append(result_dict)

    n = len(result_list); assert 0 <= label_top_k <= n
    if n == 0: return
    min_time = min(x['time'] for x in result_list)
    for result_dict in result_list:
        result_dict['time'] = (result_dict['time'] - min_time) / 86400.0

    for key in ['solved_problems', 'solved_sub_problems']:
        result_list = sorted(result_list, reverse=True, key=lambda x: x[key])
        plt.figure(figsize=(20, 8))

        _result_list = result_list[label_top_k:]
        ax = sns.scatterplot(
            x=[x['time'] for x in _result_list],
            y=[x[key] for x in _result_list],
            color='red')

        _result_list = result_list[:label_top_k]
        x = [x['time'] for x in _result_list]
        y = [x[key] for x in _result_list]
        labels = ["%s. %s (%s, %s)" % (i+1, x['name'], x['solved_sub_problems'],
            x['solved_problems']) for i, x in enumerate(_result_list)]
        ax = sns.scatterplot(x=x, y=y, color='blue')
        _scatter_label(ax, len(_result_list), x, y)

        # Customize the plot
        plt.grid(True, axis='y', color='lightgray', alpha=0.5)
        plt.title('Comparison of %s for %s SciCode Eval (Background: %s)' % \
            (key, n, with_background))
        plt.xlabel('Evaluation Date (Days)')
        plt.ylabel(key)

        if len(labels) > 0:
            plt.legend(title='Top Evaluations:\n%s' % '\n'.join(labels),
                bbox_to_anchor=(1.01, 1),
                loc='upper left')
        plt.tight_layout()

        # Save the plot to file
        out_file = os.path.join(out_dir, "scicode_BG-%s_K-%s_N-%s.png" % \
            (with_background, key, n))
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    result_dirs = ['results/evalG*', 'results/self_improve_11_24/evalG*',
        'results/self_improve_11_29/evalG*', 'results/old_results/evalG*']
    compare_scicode_evals(result_dirs, with_background=True)
    compare_scicode_evals(result_dirs, with_background=False)
    # compare_experiments_main()
    # multirun_evalplus_exp(sys.argv[1],
    #     use_true_fitness=True,
    #     eval_indv=False)
    # generate_evalplus_weights_file(sys.argv[1])
    # compare_agent_chat_stats(sys.argv[1], indv_quartile=[0.0, 1.0])
