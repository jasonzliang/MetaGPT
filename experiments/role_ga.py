from copy import deepcopy
from functools import total_ordering
import glob
import json
import logging
import os
import random
import pickle
import time

import numpy as np
from pathos.pools import ProcessPool
from ruamel.yaml import YAML

from alg_util import randomword
from alg_util import MIN_FITNESS, EPSILON, ID_LENGTH, MIN_POP_SIZE
from llm_operators import llm_mutate, llm_crossover, parse_prompt_template, \
    llm_mutate_team, llm_crossover_team
from llm_operators import DEFAULT_MAIN_ROLE
from util import get_time, sanitize_result_dict


@total_ordering
class Individual(object):
    def __init__(self, config, gen_created=None):
        self.config = config
        self.id = self._set_id(gen_created) # Ids are unique, names are not
        self.logger = logging.getLogger('role_ga')

        self.evolve_mode = self.config.get("evolve_mode", "single")
        assert self.evolve_mode in ["single", "team", "both"]
        self.mutate_rate = self.config.get("mutate_rate", 0.5)
        assert 0 <= self.mutate_rate <= 1.0
        self.llm_config = self.config.get("llm_config", {})
        self.debug_mode = self.config.get("debug_mode", False)

        self._load_initial_role()
        self.reset()

    def _load_initial_role(self):
        self.initial_main_role = self.config.get("initial_main_role",
            DEFAULT_MAIN_ROLE)
        # if self.evolve_mode in ["single", "both"]:
        _initial_main_role = os.path.join(os.path.abspath(
            os.path.dirname(__file__)), "config/%s" % self.initial_main_role)
        if os.path.exists(_initial_main_role):
            with open(_initial_main_role, "r") as f:
                self.initial_main_role = f.read()
        self.logger.info("Initial Main Role:\n%s" % self.initial_main_role)

        self.initial_team_role = self.config.get("initial_team_role", None)
        if self.evolve_mode in ["team", "both"]:
            assert os.path.exists(self.initial_team_role)
            with open(self.initial_team_role, 'r') as f:
                self.initial_team_role = json.load(f)
            self.logger.info("Initial Team Role:\n%s" % \
                self.initial_team_role)

    def _set_id(self, gen_created):
        self.gen_created = gen_created
        return "G-%s_ID-%s" % (self.gen_created, randomword(ID_LENGTH))

    def _get_sort_fitness(self):
        if self.fitness is None:
            return MIN_FITNESS
        else:
            return max(self.fitness, MIN_FITNESS)

    def __eq__(self, other):
        return self._get_sort_fitness() == other._get_sort_fitness()

    def __ne__(self, other):
        return self._get_sort_fitness() != other._get_sort_fitness()

    def __lt__(self, other):
        return self._get_sort_fitness() < other._get_sort_fitness()

    def __str__(self):
        return "[Indv] Id: %s, Fitness: %s\n%s\n%s" % (self.id, self.fitness,
            self.main_role, self.team_role)

    def _reset_roles(self):
        self.main_role = self.initial_main_role
        self.team_role = self.initial_team_role

    def _inherit_roles(self, parent):
        self.main_role = parent.main_role
        self.team_role = parent.team_role

    def reset(self):
        self._reset_roles()
        self.fitness = None
        self.true_fitness = None
        self.result_dir = None

    def create_child(self, gen_created=None):
        child = deepcopy(self)
        child.reset()
        child._inherit_roles(self)

        child.id = child._set_id(gen_created)
        # child.fitness = self.fitness
        # child.true_fitness = self.true_fitness
        # child.result_dir = self.result_dir

        return child

    def get_fitness(self, raw_fitness=False):
        if raw_fitness:
            return self.fitness
        else:
            return self._get_sort_fitness()

    def set_fitness(self, fitness):
        self.fitness = fitness
        if self.fitness is not None:
            self.fitness = max(self.fitness, MIN_FITNESS)

    def get_true_fitness(self):
        return self.true_fitness

    def set_true_fitness(self, true_fitness):
        self.true_fitness = true_fitness

    def mutate(self, mutate_rate=None):
        if mutate_rate is None: mutate_rate = self.mutate_rate
        assert 0 <= mutate_rate <= 1.0

        if self.debug_mode:
            self.main_role += randomword(ID_LENGTH)
        elif random.random() < mutate_rate:
            # print("%s: %s" % (os.getpid(), self.evolve_mode))
            if self.evolve_mode in ["single", "both"]:
                self.main_role = llm_mutate(self.main_role, self.llm_config)
            if self.evolve_mode in ["team", "both"]:
                self.team_role = llm_mutate_team(self.team_role,
                    self.llm_config)

    # def mutate2(self, n):
    #     if self.debug_mode:
    #         self.mutate()
    #     elif random.random() < mutate_rate:
    #         assert n >= 0 and self.result_dir is not None
    #         self.main_role = llm_mutate2(self.main_role, self.result_dir, n=n,
    #             llm_config=self.llm_config)
    #         if self.evolve_team_role:
    #             self.team_role = llm_mutate_team(self.team_role,
    #                 self.llm_config)

    def crossover(self, other):
        if self.debug_mode:
            self.main_role, other.role = other.role, self.main_role
        else:
            if self.evolve_mode in ["single", "both"]:
                self.main_role = llm_crossover(self.main_role, other.main_role,
                    self.llm_config)
            if self.evolve_mode in ["team", "both"]:
                self.team_role = llm_crossover_team(self.team_role,
                    other.team_role, self.llm_config)

    # def crossover2(self, others):
    #     if self.debug_mode:
    #         self.crossover(others[0])
    #     else:
    #         other_roles = [indv.main_role for indv in others]
    #         self.main_role = llm_crossover2(self.main_role, other_roles,
    #             self.llm_config)
    #         if self.evolve_team_role:
    #             self.team_role = llm_crossover_team(self.team_role,
    #                 other_roles, self.llm_config)

    def serialize(self):
        return {'id': self.id,
            'gen_created': self.gen_created,
            'fitness': self.fitness,
            'true_fitness': self.true_fitness,
            'main_role': self.main_role,
            'team_role': self.team_role,
            'result_dir': self.result_dir}

    def deserialize(self, indv_dict):
        self.id = indv_dict.get("id", self.id)
        self.gen_created = indv_dict.get("gen_created", self.gen_created)
        self.fitness = indv_dict.get("fitness", None)
        self.true_fitness = indv_dict.get("true_fitness", None)
        self.main_role = parse_prompt_template(indv_dict.get("main_role", ""))
        self.team_role = indv_dict.get("team_role", None)
        if self.team_role is not None: assert isinstance(self.team_role, dict)
        self.result_dir = indv_dict.get("result_dir", None)


class FitnessLog(object):
    def __init__(self, name, checkpoint_dir):
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.fitness_log = os.path.join(self.checkpoint_dir,
            "%s.txt" % self.name)
        with open(self.fitness_log, "a+") as f:
            f.write("# %s NEW RUN\n" % get_time(date=True, space=True))

    def update(self, gen, max_fit, mean_fit, std_fit):
        with open(self.fitness_log, "a+") as f:
            f.write("%s %d %.4f %.4f %.4f\n" % (get_time(date=True, space=True),
                gen, max_fit, mean_fit, std_fit))
            f.flush(); os.fsync(f.fileno())


class RoleEvolutionGA(object):
    def __init__(self, config, checkpoint_dir):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.logger = logging.getLogger('evolve_role')

        self.checkpoint = self.config.get("checkpoint", False)
        self.eval_cache = self.config.get("eval_cache", False)
        if self.eval_cache: self.eval_cache_fp = \
            os.path.join(self.checkpoint_dir, "eval_cache.pkl")

        self.num_gen = self.config.get("num_gen", 5)
        assert self.num_gen > 0
        self.pop_size = self.config.get("pop_size", MIN_POP_SIZE)
        assert self.pop_size > 0
        self.num_elites = self.config.get("num_elites", 1)
        assert self.num_elites < self.pop_size
        self.reevaluate_elites = self.config.get("reevaluate_elites", True)
        self.tournament_size = self.config.get("tournament_size", 2)
        assert self.tournament_size > 0
        self.init_mutate = self.config.get("init_mutate", True)
        self.n_workers = self.config.get("n_workers", 1)
        assert self.n_workers > 0
        self.indv_config = self.config.get("indv_config", {})

        # self.mutate2_n = self.config.get("mutate2_n", 3)
        # assert self.mutate2_n >= 0
        # self.crossover2_n = self.config.get("crossover2_n", 3)
        # assert self.crossover2_n >= 1

        self._reset(); _loaded_checkpoint = False

        if self.checkpoint:
            self.fitness_logs = \
                {'fitness': FitnessLog('fitness', self.checkpoint_dir),
                'true_fitness': FitnessLog('true_fitness', self.checkpoint_dir)}
            chkpt_file = self._find_latest_checkpoint()
            loaded_chkpt = self._deserialize(file_path=chkpt_file)
            assert (chkpt_file is None) == (loaded_chkpt is False)

        if self.init_mutate and not loaded_chkpt:
            self.individuals[1:] = \
                self.pool.map(self._init_mutate, self.individuals[1:])
            # [indv.mutate(mutate_rate=1.0) for indv in self.individuals[1:]]

    def get_sorted_individuals(self, individuals):
        return sorted(individuals, reverse=True)

    def _reset_pool(self):
        if hasattr(self, "pool"):
            self.pool.close(); self.pool.join(); self.pool.clear(); del self.pool
        self.pool = ProcessPool(self.n_workers)

    def _init_mutate(self, indv):
        indv.mutate(mutate_rate=1.0)
        return indv

    def _reset(self):
        self.gen = 0
        self.individuals = []
        for i in range(self.pop_size):
            individual = Individual(self.indv_config, self.gen)
            self.individuals.append(individual)
        assert self.pop_size == len(self.individuals)
        self._reset_pool()

    def _find_latest_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir,
            "checkpoint_*.yaml"))
        if len(checkpoints) == 0:
            return None
        checkpoints = sorted(checkpoints, reverse=True, key=lambda x:
            int(x.split("_")[-1].split(".")[0]))
        return checkpoints[0]

    def stop(self):
        return self.gen >= self.num_gen

    def _serialize(self, file_path):
        assert file_path is not None

        sorted_individuals = sorted(self.individuals, reverse=True)
        pop_dict = {'generation': self.gen,
            'individuals': [x.serialize() for x in sorted_individuals]}
        self.logger.info("Saving gen %s population to %s" % (self.gen,
            file_path))
        with open(file_path, 'w') as f:
            YAML().dump(sanitize_result_dict(pop_dict), f)

    def _deserialize(self, file_path=None):
        if file_path is None or not os.path.exists(file_path): return False

        self.logger.info("Loading population from %s" % file_path)
        with open(file_path, "r") as f:
            pop_dict = YAML().load(f)

        assert hasattr(self, "individuals")
        self.gen = pop_dict.get('generation', 0) + 1
        chkpt_indv = pop_dict.get('individuals', [])
        for i, individual in enumerate(self.individuals):
            if i >= len(chkpt_indv):
                return len(chkpt_indv) > 0
            else:
                individual.deserialize(chkpt_indv[i])
        return True

    def _log_population(self):
        def _log_helper(fitnesses, name):
            if len(fitnesses) == 0:
                return
            max_fit = np.max(fitnesses)
            mean_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)
            min_fit = np.min(fitnesses)
            self.logger.info("Max %s: %s" % (name, max_fit))
            self.logger.info("Mean %s: %s (%s)" % (name, mean_fit, std_fit))
            self.logger.info("Min %s: %s" % (name, min_fit))

            if self.checkpoint:
                fitness_log = self.fitness_logs.get(name)
                fitness_log.update(self.gen, max_fit, mean_fit, std_fit)

        self.logger.info("***%s Generation %s Summary***" % \
            (self.__class__.__name__, self.gen))

        fitnesses = [x.get_fitness(True) for x in self.individuals \
            if x.get_fitness(True) is not None]
        _log_helper(fitnesses, "fitness")

        true_fitnesses = [x.get_true_fitness() for x in self.individuals \
            if x.get_true_fitness() is not None]
        _log_helper(true_fitnesses, "true_fitness")

        self.logger.debug("Population:")
        for individual in self.individuals:
            self.logger.debug(individual)

    def end_gen(self):
        if self.checkpoint:
            self._serialize(os.path.join(self.checkpoint_dir,
                "checkpoint_%s.yaml" % self.gen))
        self._log_population()
        self.gen += 1

    def get_gen(self):
        return self.gen

    def _tournament_selection(self):
        chosen_ones = np.random.choice(self.individuals,
            size=min(len(self.individuals), self.tournament_size),
            replace=False)
        return np.max(chosen_ones)

    def _generate_individual_wrapper(self, idx):
        return self._generate_individual()

    def _generate_individual(self):
        parent_a = self._tournament_selection()
        parent_b = self._tournament_selection()

        child_a = parent_a.create_child(self.gen)
        child_b = parent_b.create_child(self.gen)

        child_a.crossover(child_b); child = child_a
        child.mutate()
        # assert not child.role.startswith("PROMPT_TEMPLATE: str =")
        return child

    # def _generate_individual2(self):
    #     parents = sorted([self._tournament_selection() for i in \
    #         range(self.crossover2_n + 1)], reverse=True)
    #     child = parents[0].create_child(self.gen); others = parents[1:]

    #     if random.random() < 0.5:
    #         child.crossover2(others)
    #     else:
    #         child.mutate2(self.mutate2_n)

    #     return child

    def _ask(self):
        if self.gen == 0:
            return self.individuals

        sorted_individuals = self.get_sorted_individuals(self.individuals)

        new_individuals = sorted_individuals[:self.num_elites]
        if self.n_workers == 1:
            while len(new_individuals) < self.pop_size:
                new_individuals.append(self._generate_individual2())
        else:
            created_indv = self.pool.map(self._generate_individual_wrapper,
                range(self.pop_size - self.num_elites))
            new_individuals.extend(created_indv)

        self.individuals = new_individuals
        assert self.pop_size == len(self.individuals)

        if self.reevaluate_elites:
            return self.individuals
        else:
            return self.individuals[self.num_elites:]

    # Wraper function to enable caching for ask()
    def ask(self):
        if self.eval_cache and os.path.exists(self.eval_cache_fp):
            with open(self.eval_cache_fp, "rb") as f:
                cache_indvs = pickle.load(f)
            assert len(cache_indvs) == len(self.individuals) == self.pop_size
            self.individuals = cache_indvs
            if self.reevaluate_elites: return self.individuals
            else: return self.individuals[self.num_elites:]

        eval_indvs = self._ask()
        if self.eval_cache:
            with open(self.eval_cache_fp, "wb") as f:
                pickle.dump(self.individuals, f)
        return eval_indvs

    def _check_eval_indv(self, eval_indvs):
        indv_ids = set([indv.id for indv in self.individuals])
        assert len(indv_ids) == len(self.individuals)
        for eval_indv in eval_indvs:
            assert eval_indv.id in indv_ids
            assert eval_indv in self.individuals

    def tell(self, eval_indvs, result_dicts):
        assert self.pop_size == len(self.individuals)
        assert len(eval_indvs) == len(result_dicts)
        self._check_eval_indv(eval_indvs)

        if self.reevaluate_elites:
            assert len(eval_indvs) == len(self.individuals)
        else:
            assert len(eval_indvs) == self.pop_size - self.num_elites

        for eval_indv, result_dict in \
            zip(eval_indvs, result_dicts):

            eval_indv.set_fitness(result_dict.get('fitness', None))
            eval_indv.set_true_fitness(result_dict.get('true_fitness', None))
            eval_indv.result_dir = result_dict.get('result_dir', None)

        if self.eval_cache:
            assert os.path.exists(self.eval_cache_fp)
            os.system("rm %s" % self.eval_cache_fp)
