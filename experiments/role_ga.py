import os
import glob
import json
import logging
import random
import time

import numpy as np
import ruamel.yaml as yaml


def get_time(space=True, date=True):
    '''Creates a nicely formated timestamp'''
    if date:
        date_str = "%Y-%m-%d %H:%M:%S"
    else:
        date_str = "%H:%M:%S"

    if not space:
        date_str = date_str.replace(":", "-").replace(" ", "_")

    return datetime.datetime.now(timezone('US/Pacific')).strftime(date_str)


class FitnessLog(object):
    def __init__(self, name, checkpoint_dir):
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.fitness_log = os.path.join(self.checkpoint_dir,
            "%s.txt" % self.name)
        with open(self.fitness_log, "a+") as f:
            f.write("# %s/%s NEW RUN\n" % (int(time.time()),
                get_time(date=False)))

    def update(self, gen, max_fitness, mean_fitness):
        with open(self.fitness_log, "a+") as f:
            f.write("%s/%s %s %s %s\n" % \
                (int(time.time()), get_time(date=False), gen, max_fitness,
                    mean_fitness))
            f.flush()
            os.fsync(f.fileno())


@total_ordering
class Individual(object):
    def __init__(self, config, gen_created=None):
        self.config = config
        self.id = self._set_id(gen_created) # Ids are unique, names are not
        self.reset()

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
        return "[Indv] Id: %s, Fitness: %s\n%s" % (self.id, self.fitness,
            self.role)

    def _reset_roles(self):
        self.role = ""

    def _inherit_roles(self, parent):
        self.role = deepcopy(parent.role)

    def reset(self):
        self._reset_roles()
        self.fitness = None
        self.true_fitness = None

    def create_child(self, gen_created=None):
        child = deepcopy(self)
        child.reset()
        child._inherit_roles(self)

        child.id = child._set_id(gen_created)
        child.fitness = self.fitness
        child.true_fitness = self.true_fitness

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

    # TODO
    def mutate(self):
        pass

    # TODO
    def crossover(self, other):
        pass

    def serialize(self):
        return {'id': self.id,
            'gen_created': self.gen_created,
            'fitness': self.fitness,
            'true_fitness': self.true_fitness
            'role': self.role}

    def deserialize(self, indv_dict):
        self.id = indv_dict.get("id", self.id)
        self.gen_created = indv_dict.get("gen_created", self.gen_created)
        self.fitness = indv_dict.get("fitness", None)
        self.true_fitness = indv_dict.get("true_fitness", None)

        self.role = indv_dict.get("role", "")


class RoleEvolutionGA(object):
    def __init__(self, config, checkpoint_dir):
        self.config = config
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.logger = logging.getLogger('root')

        self.checkpoint = self.config.get("checkpoint", False)
        self.pop_size = self.config.get("pop_size", MIN_POP_SIZE)
        self.num_gen = self.config.get("num_gen", 5)
        self.num_elites = self.config.get("num_elites", 1)
        self.reevaluate_elites = self.config.get("reevaluate_elites", True)

        assert self.num_gen > 0

        if self.checkpoint:
            self.fitness_logs = \
                {'fitness': FitnessLog('fitness', self.checkpoint_dir),
                'true_fitness': FitnessLog('true_fitness', self.checkpoint_dir)}
            chkpt_file = self._find_latest_checkpoint()
        else:
            chkpt_file = None

        if chkpt_file is not None:
            self._deserialize(file_path=chkpt_file)
        else:
            self._reset()

    def get_sorted_individuals(self, individuals):
        return sorted(individuals, reverse=True)

    def _reset(self):
        self.gen = 0
        self.individuals = []
        for i in range(self.pop_size):
            individual = Individual(self.gene_configs, self.gen)
            self.individuals.append(individual)
        assert len(self.individuals) == self.pop_size

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
            yaml.dump(sanitize_result_dict(pop_dict), f)

    def _deserialize(self, pop_dict=None, file_path=None):
        assert bool(pop_dict) != bool(file_path)
        if file_path is not None:
            assert pop_dict is None
            self.logger.info("Loading population from %s" % file_path)
            with open(file_path, "r") as f:
                pop_dict = yaml.load(f)
        else:
            assert pop_dict is not None

        if not hasattr(self, "individuals"):
            self._reset()
        self.gen = pop_dict.get('generation', 0) + 1
        new_indv = pop_dict.get('individuals', [])
        for i, individual in enumerate(self.individuals):
            if i >= len(new_indv):
                return
            else:
                individual.deserialize(new_indv[i])

    def _log_population(self):
        def _log_helper(fitnesses, name):
            if len(fitnesses) == 0:
                return
            max_fitness = np.max(fitnesses)
            mean_fitness = np.mean(fitnesses)
            min_fitness = np.min(fitnesses)
            self.logger.info("Max %s: %s" % (name, max_fitness))
            self.logger.info("Mean %s: %s" % (name, mean_fitness))
            self.logger.info("Min %s: %s" % (name, min_fitness))

            if self.checkpoint:
                fitness_log = self.fitness_logs.get(name)
                fitness_log.update(self.gen, max_fitness, mean_fitness)

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

    def _generate_individual(self):
        counter = MAX_FILTER_TRIES
        while True:
            parent_a = self._tournament_selection()
            parent_b = self._tournament_selection()

            child_a = parent_a.create_child(self.gen)
            child_b = parent_b.create_child(self.gen)

            child_a.crossover(child_b)
            child = random.choice([child_a, child_b])
            child.mutate()

            if counter == 0 or self._check_indv_filters(child):
                return child
            counter -= 1

    def ask(self):
        if self.gen == 0:
            return self.individuals

        sorted_individuals = self.get_sorted_individuals(self.individuals)

        new_individuals = sorted_individuals[:self.num_elites]
        while len(new_individuals) < self.pop_size:
            new_individuals.append(self._generate_individual())

        self.individuals = new_individuals
        assert len(self.individuals) == self.pop_size

        if not self.reevaluate_elites:
            return self.individuals[self.num_elites:]
        else:
            return self.individuals

    def tell(self, eval_indvs, fitnesses, true_fitnesses):
        assert self.pop_size == len(self.individuals)
        assert len(eval_indvs) == len(true_fitnesses)
        assert len(eval_indvs) == len(fitnesses)
        assert len(eval_indvs) == len(self.individuals)

        for eval_indv, fitness, true_fitness in \
            zip(eval_indvs, fitnesses, true_fitnesses):

            eval_indv.set_fitness(fitness)
            eval_indv.set_true_fitness(true_fitness)
