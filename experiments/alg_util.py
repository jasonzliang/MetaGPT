import copy
from collections import defaultdict, abc
# import collections.abc as abc
import functools
import hashlib
import importlib
import os
import random
import string

import numpy as np
import networkx as nx

MAX_CROWDING_DIST = 1.0
ID_LENGTH = 12
EPSILON = 1e-14
MIN_POP_SIZE = 2
MIN_FITNESS = -1e12
# MIN_FITNESS = 1e-12

def moving_average(arr, n):
    avg = np.zeros(len(arr))
    for i in range(len(arr)):
        # print(i, i-n+1, i+1)
        # print(arr[max(i-n+1, 0):i+1])
        avg[i] = np.mean(arr[max(i-n+1, 0):i+1])
    return avg

def exponential_smoothing(series, alpha):
    results = np.zeros_like(series)
    results[0] = series[0]
    for t in range(1, series.shape[0]):
        results[t] = alpha * series[t] + (1 - alpha) * results[t - 1]
    return results

def recursive_update(d, u):
    if u is None: return d

    for k, v in u.items():
        if isinstance(v, abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def remove_duplicate(my_list, key_func):
    temp_dict = {}
    for l in my_list:
        temp_dict[key_func(l)] = l
    return list(temp_dict.values())

def randomword(length, seed=None):
    letters = string.ascii_uppercase + string.ascii_lowercase + string.digits
    if seed is not None: state = random.getstate(); random.seed(seed)
    rv = ''.join(random.choice(letters) for i in range(length))
    if seed is not None: random.setstate(state)
    return rv

def is_numpy_type(obj):
    return type(obj).__module__ == np.__name__

def get_arch_id(model_fp, return_index=False):
    '''Returns the architecture id and index if necessary'''
    model_name = os.path.basename(model_fp)
    for i, field in enumerate(model_name.split("_")):
        if field.startswith("A-"):
            arch_id = field.split("-")[1]
            if return_index:
                return i, arch_id
            else:
                return arch_id
    print("Warning: cannot find arch id!")
    return None, None if return_index else None

@functools.lru_cache(maxsize=None, typed=False)
def hash_str(obj_str, n):
    if type(obj_str) is not str:
        obj_str = str(obj_str)
    return hashlib.md5(obj_str.encode('utf-8')).hexdigest()[:n]

def substite_arch_id(model_fp, model):
    index, arch_id = get_arch_id(model_fp, True)
    if index is None:
        return model_fp

    new_arch_id = hash_str(model.to_json(), ID_LENGTH)

    model_name = os.path.basename(model_fp)
    model_dir = os.path.dirname(model_fp)

    fields = model_name.split("_")
    fields[index] = "A-%s" % new_arch_id
    new_model_fp = os.path.join(model_dir, "_".join(fields))
    return new_model_fp

def import_domain(pbt_dir, domain_module):
    if os.path.exists(domain_module):
        print("Converting domain file path to module")
        domain_module = os.path.splitext(os.path.expanduser(domain_module))[0]
        domain_module = domain_module.replace(pbt_dir, "").replace("/", ".")
    else:
        domain_module = domain_module

    print("Importing domain module: %s" % domain_module)
    return importlib.import_module(domain_module)

def import_tensorflow(cpu_only=False):
    # Disable excessive logging by TF
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # os.environ['TF_CUDNN_DETERMINISTIC']='1'

    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

    # Option to enable XLA
    if 'TF_ENABLE_JIT' in os.environ and os.environ['TF_ENABLE_JIT'] == '1':
        tf.config.optimizer.set_jit("autoclustering")

    # Use CPU only option
    if cpu_only:
        tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                #### Hack needed to prevent deadlock ####
                # tf.config.experimental.set_virtual_device_configuration(gpu,
                #     [tf.config.experimental.VirtualDeviceConfiguration(
                #     memory_limit=5000)])
                tf.config.experimental.set_memory_growth(gpu, True)

            # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), \
            #     "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            raise

    return tf

    ### Legacy code
    # if '_session' in globals():
    #     return
    # global _session
    # from keras import backend as K
    # _session = K.get_session()

    # _session = tf.Session(config=config)
    # # try:
    # #     K.clear_session()
    # # except:
    # #   pass
    # K.set_session(_session)

def product(values):
    if len(values) == 0:
        return 0
    else:
        return functools.reduce((lambda x, y: x * y), values)

def list_diff(my_list):
    assert len(my_list) > 1
    return [j - i for i, j in zip(my_list[: -1], my_list[1 :])]

def get_source_sink_nodes(dag_graph, data=False):
    sources = list(dag_graph.nodes(data=data))
    sinks = list(dag_graph.nodes(data=data))
    # print(sources)
    for u, v in dag_graph.edges():
        if u in sinks:
            sinks.remove(u)
        if v in sources:
            sources.remove(v)
    return sources, sinks

def bfs_traversal(dag_graph, source):
    queue = [source]
    traversal = []

    while len(queue) > 0:
        node = queue.pop(0)
        traversal.append(node)
        for u, v in dag_graph.out_edges(node):
            if v not in queue:
                queue.append(v)

    # print(traversal)
    return traversal

def replace_node_with_graph(dag_graph, node, sub_graph):
    in_nodes = []
    for (u, v) in dag_graph.in_edges(node):
        in_nodes.append(u)
    out_nodes = []
    for (u, v) in dag_graph.out_edges(node):
        out_nodes.append(v)

    joint_graph = nx.compose(dag_graph, sub_graph)
    joint_graph.remove_node(node)
    sub_sources, sub_sinks = get_source_sink_nodes(sub_graph)

    for u in in_nodes:
        for v in sub_sources:
            joint_graph.add_edge(u, v)
    for u in sub_sinks:
        for v in out_nodes:
            joint_graph.add_edge(u, v)

    assert nx.is_weakly_connected(joint_graph)
    return joint_graph

def insert_subgraph_after_node(dag_graph, node, sub_graph):
    out_nodes = []
    for (u, v) in dag_graph.out_edges(node):
        out_nodes.append(v)

    joint_graph = nx.compose(dag_graph, sub_graph)
    sub_sources, sub_sinks = get_source_sink_nodes(sub_graph)

    for v in sub_sources:
        joint_graph.add_edge(node, v)
    for u in sub_sinks:
        for v in out_nodes:
            joint_graph.add_edge(u, v)

    assert nx.is_weakly_connected(joint_graph)
    return joint_graph

def relabel_graph(nx_graph, n=ID_LENGTH):
    nodes = list(nx_graph.nodes())
    new_nodes = [randomword(n) for x in nodes]
    mapping = dict(zip(nodes, new_nodes))
    return nx.relabel_nodes(nx_graph, mapping)

def join_two_graphs_serial(graph_1, graph_2):
    sources1, sinks1 = get_source_sink_nodes(graph_1)
    sources2, sinks2 = get_source_sink_nodes(graph_2)

    joint_graph = nx.compose(graph_1, graph_2)

    for u in sinks1:
        for v in sources2:
            joint_graph.add_edge(u, v)

    assert nx.is_weakly_connected(joint_graph)
    return joint_graph

def join_two_graphs_parallel(graph_1, graph_2, new_source, new_sink):
    sources1, sinks1 = get_source_sink_nodes(graph_1)
    sources2, sinks2 = get_source_sink_nodes(graph_2)

    joint_graph = nx.compose(graph_1, graph_2)

    for u in sources1 + sources2:
        joint_graph.add_edge(new_source, u)
    for u in sinks1 + sinks2:
        joint_graph.add_edge(u, new_sink)

    assert nx.is_weakly_connected(joint_graph)
    return joint_graph

def get_secondary_obj(indv, sec_obj_name):
    if type(sec_obj_name) is dict:
        return sec_obj_name[indv.id]
    else:
        assert type(sec_obj_name) is str
        return indv.get_secondary_obj(sec_obj_name)

def single_pareto_front(individuals, sec_obj_name, max_obj_1, max_obj_2):

    sorted_indv = sorted(individuals, reverse=max_obj_1)
    p_front = [sorted_indv[0]]

    for indv in sorted_indv[1:]:
        if max_obj_2:
            if get_secondary_obj(indv, sec_obj_name) >= \
                get_secondary_obj(p_front[-1], sec_obj_name):
                p_front.append(indv)

        else:
            if get_secondary_obj(indv, sec_obj_name) <= \
                get_secondary_obj(p_front[-1], sec_obj_name):
                p_front.append(indv)
    return p_front

def get_crowding_dist(p_front, sec_obj_name):
    crowding_dist_dict = defaultdict(float)

    p_front = sorted(p_front, reverse=False)
    min_value = p_front[0].get_fitness()
    max_value = p_front[-1].get_fitness()

    for i, indv in enumerate(p_front):
        if i == 0 or i == len(p_front) - 1:
            dist = MAX_CROWDING_DIST
        else:
            after = p_front[i + 1].get_fitness()
            before = p_front[i - 1].get_fitness()
            dist = float(after - before) / ((max_value - min_value) + EPSILON)
            assert dist <= 1.0

        crowding_dist_dict[indv.id] += dist

    p_front = sorted(p_front, key=lambda x: get_secondary_obj(x, sec_obj_name),
        reverse=False)
    min_value = get_secondary_obj(p_front[0], sec_obj_name)
    max_value = get_secondary_obj(p_front[-1], sec_obj_name)

    for i, indv in enumerate(p_front):
        if i == 0 or i == len(p_front) - 1:
            dist = MAX_CROWDING_DIST
        else:
            after = get_secondary_obj(p_front[i + 1], sec_obj_name)
            before = get_secondary_obj(p_front[i - 1], sec_obj_name)
            dist = float(after - before) / ((max_value - min_value) + EPSILON)
            assert dist <= 1.0

        crowding_dist_dict[indv.id] += dist

    return crowding_dist_dict

def nd_sort(individuals, sec_obj_name, reverse=False, max_obj_1=True,
    max_obj_2=True):

    _individuals = copy.copy(individuals)

    p_front_dict = {}
    p_front_counter = 0
    while len(_individuals) > 0:
        p_front = single_pareto_front(_individuals, sec_obj_name, max_obj_1,
            max_obj_2)
        crowding_dist_dict = get_crowding_dist(p_front, sec_obj_name)
        # print([indv.id for indv in p_front])

        for indv in p_front:
            p_front_dict[indv.id] = (p_front_counter,
                crowding_dist_dict[indv.id])

        _new_individuals = []
        p_front_ids = [indv.id for indv in p_front]
        for indv in _individuals:
            if indv.id not in p_front_ids:
                _new_individuals.append(indv)
        _individuals = _new_individuals

        p_front_counter -= 1

    # if shuffle_pfront:
    #     individuals = random.sample(individuals, len(individuals))
    #     sorted_individuals = sorted(individuals,
    #         key=lambda x: p_front_dict[x.id][0], reverse=reverse)
    # else:
    sorted_individuals = sorted(individuals,
        key=lambda x: p_front_dict[x.id], reverse=reverse)

    return sorted_individuals, p_front_dict

def merge_keras_history(past_hist, hist):
    if len(hist) == 0:
        return past_hist
    new_hist = {}
    for key in hist:
        if key in past_hist:
            new_hist[key] = past_hist[key] + hist[key]
        else:
            new_hist[key] = hist[key]
    return new_hist
