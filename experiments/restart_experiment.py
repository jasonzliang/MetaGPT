#!/usr/bin/env python

from datetime import datetime
import glob
import os
import signal
import sys
import subprocess
import time

from ruamel.yaml import YAML
import numpy as np
import psutil

from util import randomword, get_time

SLEEP_TIME = 30 # Interval to check whether restart conditions meet
WAIT_TIME = 3 # Grace period after killing an experiment process
RESTART_TIME = 120 # Maximum time to wait after launching new experiment
MAX_MEMORY_PERCENT = 100.0 # Maximum memory usage before restarting experiment
MAXIMUM_RESTARTS = 10 # Maximum number of times restarts can happen
REMAINING_GPU_PROC_CHECKS = MAX_GPU_PROC_CHECKS

DEFAULT_SCREEN_NAME = "experiment" # Default name of screen to use
EXP_BASE_DIR = "~/Desktop/MetaGPT/experiments" # Base directory for running experiments
EXP_SCRIPT_NAME = "evolve_role.py" # Name of main experiment script
SCROLLBACK_LINES = 10000000 # number of lines to set scrollback"\

MAX_GEN_TIMEOUT = 50000 # Maximum generation timeout
ADAPTIVE_GEN_TIMEOUT = True # Change gen timeout depending on past gen lengths
MIN_DATA_PTS = 10 # Minimum data points for adaptive timeout to be enabled
USE_LATEST_PTS = None # Use only the latest data points to determine timeout
CALC_METHOD = 'max' # Either mean, median, max for determining typical gen time
STD_THRESHOLD = 4.0 # Timeout is this many stds above the typical gen time
TIMEOUT_MULTIPLIER = 1.5 # Multiplier to the timeout, use if higher than STD
RESTART_LOG = 'restart.log' # Log file for writing reason for restart

class RestartException(Exception):
    pass

def get_checkpoints(experiment_dir, is_base_dir=True, min_gen=None,
    max_gen=None, verbose=True):
    if is_base_dir:
        checkpoints = glob.glob(os.path.join(experiment_dir,
            "role_ga/*.yaml"))
    else:
        checkpoints = glob.glob(os.path.join(experiment_dir,
            "*.yaml"))
    checkpoints = sorted(checkpoints,
        key=lambda x: int(x.rstrip(".yaml").split("_")[-1]))

    if min_gen is None or min_gen < 0:
        min_gen = 0
    if max_gen is None or max_gen > len(checkpoints):
        max_gen = len(checkpoints)

    within_range = []
    for checkpoint in checkpoints:
        n = int(checkpoint.rstrip(".yaml").split("_")[-1])
        if n >= min_gen and n < max_gen:
            within_range.append(checkpoint)
    if verbose:
        print("Retrieved %s checkpoints" % len(within_range))
    return within_range

def update_scrollback():
    screen_file = os.path.expanduser("~/.screenrc")
    if not os.path.exists(screen_file):
        bash_command("touch %s" % screen_file)
    with open(screen_file, 'r') as f:
        if "defscrollback" in f.read():
            return
    with open(screen_file, 'a') as f:
        f.write("defscrollback %s" % SCROLLBACK_LINES)
        f.flush()
        os.fsync(f.fileno())

def kill_proc_tree(pid, experiment_name, sig=signal.SIGTERM,
    include_parent=True, timeout=WAIT_TIME, on_terminate=None):
    """Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callabck function which is
    called as soon as a child terminates.
    """
    try:
        assert pid != os.getpid(), "won't kill myself"
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        if include_parent:
            children.append(parent)
        for p in children:
            p.send_signal(sig)
        gone, alive = psutil.wait_procs(children, timeout=timeout,
            callback=on_terminate)
        if len(alive) > 0:
            return False

        print("Killed timed out experiment [%s] with pid %s" %
            (experiment_name, pid))
        return True

    except psutil.NoSuchProcess as e:
        print("No such process: %s" % pid)
        return True

def bash_command(cmd):
    process = subprocess.Popen(cmd, shell=True, executable='/bin/bash',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = process.communicate  ()#; print(out)
    return out

def check_screen_exists(screen_name):
    try:
        result = subprocess.check_output("screen -ls", shell=True)
        result = result.decode("ascii")
    except:
        result = None

    if result is not None and screen_name in result:
        return True
    else:
        return False

def get_exp_name_cmdline(cmdline):
    try:
        args = cmdline.split(EXP_SCRIPT_NAME)[-1]
        exp_path = [x.strip() for x in args.split()][0]
        exp_name = os.path.basename(exp_path)
    except:
        exp_name = cmdline
    return exp_name

def restart_experiment(directory, config_file, reason=None):
    assert os.path.exists(config_file)
    experiment_name = os.path.basename(directory)

    while True:
        can_break = True
        proc_iter = psutil.process_iter(attrs=["pid", "name", "cmdline"])

        for p in proc_iter:
            cmdline = " ".join(p.cmdline())

            if EXP_SCRIPT_NAME in cmdline: # and experiment_name in cmdline:
                kill_exp_name = get_exp_name_cmdline(cmdline)
                killed_process = kill_proc_tree(p.pid, kill_exp_name)
                print("Experiment [%s] killed" % kill_exp_name)
                if not killed_process:
                    can_break = False
        if can_break:
            break

    if not check_screen_exists(DEFAULT_SCREEN_NAME):
        bash_command("screen -dmS %s" % DEFAULT_SCREEN_NAME)
        print("Experiment screen [%s] is (re)started" % DEFAULT_SCREEN_NAME)

    time.sleep(WAIT_TIME)
    script_cmd = './%s %s %s' % (EXP_SCRIPT_NAME, directory, config_file)
    screen_cmd = "screen -S %s -X stuff 'cd %s; %s'$(echo -ne '\015')" % \
        (DEFAULT_SCREEN_NAME, EXP_BASE_DIR, script_cmd)
    print("(Re)starting experiment [%s] on screen [%s] with cmd: %s" % \
        (experiment_name, DEFAULT_SCREEN_NAME, script_cmd))
    bash_command(screen_cmd)

    total_wait_time = RESTART_TIME
    log_file = os.path.join(directory, "server.log")
    while not os.path.exists(log_file):
        if total_wait_time <= 0:
            break
        total_wait_time -= WAIT_TIME
        time.sleep(WAIT_TIME)
    assert os.path.exists(log_file)

    global MAXIMUM_RESTARTS; MAXIMUM_RESTARTS -= 1
        print("No more (re)starts remaining, exiting")
    if MAXIMUM_RESTARTS == 0:
        raise RestartException
    else:
        print("Experiment (re)start successful, %s remaining" %
            MAXIMUM_RESTARTS)
    with open(os.path.join(directory, RESTART_LOG), 'a') as f:
        f.write("[%s][%s] %s\n" % (get_time(), MAXIMUM_RESTARTS, reason))

def server_log_timeout(log_file, timeout):
    assert os.path.exists(log_file)
    time_elapsed = time.time() - os.path.getmtime(log_file)

    if time_elapsed > timeout:
        print("Server log file timeout of %d > %d seconds reached" % \
            (time_elapsed, timeout))
        return True
    else:
        return False


def calculate_adaptive_timeout(directory, base_timeout):
    if not ADAPTIVE_GEN_TIMEOUT:
        return base_timeout
    fitness_file = os.path.join(directory, "role_ga/fitness.txt")
    if not os.path.exists(fitness_file):
        return base_timeout

    gen_times = []
    timestamps = []
    with open(fitness_file, 'r') as f:
        for line in f.readlines() + ["#"]:
            if line.startswith("#"):
                if len(timestamps) >= 2:
                    gen_times += [t - s for s, t in zip(
                        timestamps, timestamps[1:])]
                timestamps = []
                continue
            t, g, b, m = line.rstrip().split()
            t = t.split('/')[0]
            timestamps.append(float(t))

    assert MIN_DATA_PTS is not None
    if len(gen_times) < MIN_DATA_PTS:
        return base_timeout
    if USE_LATEST_PTS is not None:
        gen_times = gen_times[-USE_LATEST_PTS:]

    if CALC_METHOD == "mean":
        calc_func = np.mean
    elif CALC_METHOD == "median":
        calc_func = np.median
    elif CALC_METHOD == "max":
        calc_func = np.max
    else:
        raise ValueError("Invalid CALC_METHOD: %s" % CALC_METHOD)

    adaptive_timeout = max(float(calc_func(gen_times)) * TIMEOUT_MULTIPLIER,
        float(calc_func(gen_times) + STD_THRESHOLD*np.std(gen_times)))

    if adaptive_timeout > base_timeout or adaptive_timeout < SLEEP_TIME:
        print("WARNING: Adaptive timeout (%d) outside acceptable limits" % \
            adaptive_timeout)
    return min(max(SLEEP_TIME, adaptive_timeout), base_timeout)

def generation_timeout(directory, monitor_start_time, timeout):
    checkpoints = get_checkpoints(directory, verbose=False)
    if len(checkpoints) == 0:
        return False

    last_checkpoint = checkpoints[-1]
    last_checkpoint_name = os.path.basename(last_checkpoint)
    time_elapsed = time.time() - max(monitor_start_time,
        os.path.getmtime(last_checkpoint))
    timeout = calculate_adaptive_timeout(directory, timeout)

    print("[%s] %d/%d seconds since last checkpoint (%s) is updated" % \
        (get_time(), time_elapsed, timeout, last_checkpoint_name))

    if time_elapsed > timeout:
        print("Generation timeout of %d > %d seconds reached" % \
            (time_elapsed, timeout))
        return True
    else:
        return False

def not_enough_free_memory():
    mem = psutil.virtual_memory()

    if mem.percent > MAX_MEMORY_PERCENT:
        print("Maximum memory usage of %.1f%% > %.1f%% reached" % \
            (mem.percent, MAX_MEMORY_PERCENT))
        return True
    else:
        return False

def monitor_experiment(directory, config_file=None, timeout=MAX_GEN_TIMEOUT,
    gen_timeout=MAX_GEN_TIMEOUT):

    assert timeout >= 1
    assert gen_timeout >= 1
    global SLEEP_TIME
    SLEEP_TIME = min(timeout, SLEEP_TIME)

    experiment_name = os.path.basename(directory)
    log_file = os.path.join(directory, "server.log")
    if config_file is None:
        config_file = os.path.join(directory, "config.yaml")
    assert os.path.exists(config_file)

    update_scrollback()
    monitor_start_time = time.time()

    while True:
        if os.path.exists(os.path.join(directory, "done")):
            print("Experiment finished, monitoring stopped")
            return True

        if not os.path.exists(directory) or \
            not check_screen_exists(DEFAULT_SCREEN_NAME):
            restart_experiment(directory, config_file, "initial startup")

        assert os.path.exists(config_file)
        assert os.path.exists(log_file)

        print("[%s] %d/%d seconds since log for experiment [%s] updated" % \
            (get_time(), int(time.time() - os.path.getmtime(log_file)), timeout,
                experiment_name))

        restart_reason = None
        if server_log_timeout(log_file, timeout):
            restart_reason = "server log timeout"
        if generation_timeout(directory, monitor_start_time, gen_timeout):
            restart_reason = "generation timeout"
        if not_enough_free_memory():
            restart_reason = "memory timeout"

        if restart_reason is not None:
            restart_experiment(directory, config_file, restart_reason)
            monitor_start_time = time.time()

        time.sleep(SLEEP_TIME)

def check_config(batch_config):
    experiment_dirs = set([])
    for i, experiment_list in enumerate(batch_config.get("experiments", [])):
        assert len(experiment_list) == 4:
        experiment_dir, experiment_config, log_timeout, server_timeout = \
            experiment_list

        assert experiment_dir not in experiment_dirs
        assert experiment_config is None or os.path.exists(experiment_config)
        assert log_timeout > 0
        assert server_timeout > 0
        experiment_dirs.add(experiment_dir)

def monitor_experiment_batch(config_file):
    assert os.path.exists(config_file)
    with open(config_file, 'r') as f:
        batch_config = YAML().load(f)
    add_serial = batch_config.get('serial', False)
    add_hostid = batch_config.get("hostid", False)
    add_timestamp = batch_config.get("timestamp", False)
    git_pull = batch_config.get("git_pull", False)

    if git_pull:
        result = bash_command("git pull").decode('utf-8').rstrip()
        if 'Already up to date' not in result \
            and 'Already up-to-date' not in result:
            print("Restarting monitoring script due to git updates")
            os.execv(__file__, sys.argv)
            return

    check_config(batch_config)
    global MAXIMUM_RESTARTS
    maximum_restarts = MAXIMUM_RESTARTS
    for i, experiment_list in enumerate(batch_config.get("experiments", [])):
        assert len(experiment_list) == 4
        experiment_dir, experiment_config, log_timeout, server_timeout = \
            experiment_list

        if add_serial:
            experiment_dir += "_v%s" % (i + 1)
        if add_hostid:
            hostid = bash_command("hostid").decode("utf-8").rstrip()
            experiment_dir += "_%s" % hostid
        if add_timestamp:
            experiment_dir += "_%s" % int(time.time())

        print("\nMonitoring exp [%s] with config [%s] and timeouts %s/%s\n" %
            (experiment_dir, experiment_config, log_timeout, server_timeout))
        MAXIMUM_RESTARTS = maximum_restarts
        try:
            monitor_experiment(experiment_dir, experiment_config, log_timeout,
                server_timeout)
        except RestartException:
            print("Too many (re)starts, skipping to next experiment")

if __name__ == '__main__':
    if len(sys.argv) == 2:
        monitor_experiment_batch(sys.argv[1])
    elif len(sys.argv) == 3:
        monitor_experiment(sys.argv[1], timeout=int(sys.argv[2]))
    elif len(sys.argv) == 4:
        monitor_experiment(sys.argv[1], timeout=int(sys.argv[2]),
            gen_timeout=int(sys.argv[3]))
    else:
        print("Usage options:")
        print("1) ./restart_experiment.py " \
            "[experiment directory][log timeout duration]" \
            "[(optional) gen timeout duration]")
        print("2) ./restart_experiment.py [batch config file]")
