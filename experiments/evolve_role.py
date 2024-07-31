#!/usr/bin/env python
import os
import sys
# Hack to avoid adding pbt to PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(os.path.realpath(__file__))))

import filecmp
import random
import shutil
import traceback
import time

import numpy as np
from ruamel.yaml import YAML

from logger import setup_experiment_logging, log_results
from llm_evaluator import LLMEvaluator
from role_ga import RoleEvolutionGA
from util import get_time, sanitize_result_dict

LINE_WIDTH = 80
MAX_PATH_LENGTH = 256


class RoleEvolutionServer(object):
    def __init__(self, experiment_dir, config_file=None):
        # Setup experiment directory
        self.experiment_dir = experiment_dir
        resume = False
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            resume = True

        # Load configuration file
        self.config_file = config_file
        if self.config_file is None:
            self.config_file = os.path.join(self.experiment_dir, "config.yaml")
        assert os.path.exists(self.config_file)
        with open(self.config_file, "rb") as f:
            self.config = YAML().load(f)
        if self.config is None:
            self.config = {}

        # Setup logger
        self.logger = setup_experiment_logging('evolve_role',
            os.path.join(self.experiment_dir, "server.log"),
            self.config.get('verbosity', 'info'))
        self.logger.info("Loaded config file: %s" % self.config_file)
        if resume:
            self.logger.info("Resuming existing experiment: %s" % \
                self.experiment_dir)
        else:
            self.logger.info("Starting new experiment: %s" % \
                self.experiment_dir)

        # Write config file to experiment dir
        exp_config_fp = os.path.join(self.experiment_dir, "config.yaml")
        if os.path.exists(exp_config_fp) and \
            not filecmp.cmp(self.config_file, exp_config_fp):
            os.rename(exp_config_fp, "%s.%s" % (exp_config_fp,
                int(os.path.getmtime(exp_config_fp))))
            assert not os.path.exists(exp_config_fp)

        if os.path.realpath(os.path.expanduser(self.config_file)) \
            != os.path.realpath(os.path.expanduser(exp_config_fp)):
            shutil.copy(self.config_file, exp_config_fp)

        # Setup GA and evaluator
        self.ga = RoleEvolutionGA(self.config.get("role_ga_config", {}),
            os.path.join(self.experiment_dir, "role_ga"))
        self.evaluator = LLMEvaluator(
            self.config.get("llm_evaluator_config", {}),
            os.path.join(self.experiment_dir, "llm_evaluator"))

        # Wrap code to catch error and log it to file
        try:
            self.run()
        except Exception as e:
            error = str(e).replace(" ", "_").replace("'", "").replace('"', '')\
                .replace("\\", "_")
            filename = "%s_%s.err" % (int(time.time()), error)
            filename = filename[:MAX_PATH_LENGTH]
            with open(os.path.join(self.experiment_dir, filename), 'w') as f:
                f.write(traceback.format_exc())
            raise e

    def run(self):
        self.logger.info("*" * LINE_WIDTH)
        self.logger.info("STARTING EXPERIMENT RUN")
        self.logger.info("%s" % get_time())
        self.logger.info("*" * LINE_WIDTH)

        experiment_done = os.path.join(self.experiment_dir, "done")
        if os.path.exists(experiment_done): os.remove(experiment_done)

        while not self.ga.stop():
            ga_gen = self.ga.get_gen()
            self.logger.info("*** START OF GENERATION %s ***" % ga_gen)
            population = self.ga.ask()

            result_dicts = self.evaluator.evaluate(population)
            result_dicts = sanitize_result_dict(result_dicts)
            assert len(result_dicts) == len(population)

            self.ga.tell(population, result_dicts)
            self.logger.debug("Evaluated results: %s" % result_dicts)

            self.ga.end_gen()
            self.logger.info("***END OF GENERATION***\n\n")

        self.logger.info("*" * LINE_WIDTH)
        self.logger.info("Experiment run finished")
        self.logger.info("*" * LINE_WIDTH)

        if self.ga.stop():
            os.system("touch %s" % experiment_done)
            os.system("rm -rf /tmp/*")

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: ./main.py [experiment directory] [config file]")
    elif len(sys.argv) == 2:
        RoleEvolutionServer(sys.argv[1])
    else:
        RoleEvolutionServer(sys.argv[1], sys.argv[2])
