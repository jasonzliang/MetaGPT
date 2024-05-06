import logging
import ruamel.yaml as yaml

from collections import OrderedDict
# import coloredlogs
# coloredlogs.install()
LOGGERS = {}
VERBOSITY_MAP = {'debug': logging.DEBUG, 'info': logging.INFO}
FMT = '%(asctime)s - %(process)d - %(levelname)s - %(module)s - %(message)s'
DATEFMT = '%H:%M:%S'

def setup_experiment_logging(name, log_file=None, verbosity='debug'):
    global LOGGERS
    if name in LOGGERS:
        logger = LOGGERS.get(name)
        logger.debug("Found existing logger: %s" % name)
        return logger

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(VERBOSITY_MAP.get(verbosity, logging.INFO))

    formatter = logging.Formatter(fmt=FMT, datefmt=DATEFMT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a+')
        logger.addHandler(file_handler)

    LOGGERS[name] = logger

    logger.debug("Created new logger: %s" % name)
    return logger

def log_results(result_dicts, file_name):
    # for result_dict, indv in zip(result_dicts, population):
    #     result_dict['id'] = indv.id

    result_dicts = sorted(result_dicts, reverse=True,
        key=lambda x: x.get("fitness"))
    # id_result_list = zip([x.id for x in population], result_dicts)
    # id_result_dict = OrderedDict(sorted(id_result_list, reverse=True,
    #     lambda x: x[1].get("fitness", None)))

    with open(file_name, "w") as f:
        yaml.dump(result_dicts, f)
