#!/usr/bin/env python
import os
import sys
import time

SERVER_IP = "biggpu"
EXPERIMENT_NAME = "*"
SERVER_DIR = "~/Desktop/MetaGPT/experiments/results"
EXCLUDE_DIRS = []
LOCAL_DIR = SERVER_DIR

if __name__ == "__main__":
    print("Usage: ./download.py [exp_name]")
    print("Usage: ./download.py [exp_name] [server_ip]")

    if len(sys.argv) == 1:
        print("Error: not enough arguments!"); exit()
    elif len(sys.argv) == 2:
        EXPERIMENT_NAME = sys.argv[1]
    elif len(sys.argv) == 3:
        EXPERIMENT_NAME = sys.argv[1]
        SERVER_IP = sys.argv[2]
    else:
        print("Error: too many arguments!"); exit()

    if not EXPERIMENT_NAME.endswith("*"): EXPERIMENT_NAME += "*"
    EXCLUDE_DIRS = " ".join(["--exclude '%s'" % x for x in EXCLUDE_DIRS])
    SERVER_DIR = os.path.join(SERVER_DIR, EXPERIMENT_NAME)

    rsync_cmd = "rsync -ahvzAPX --no-i-r --stats %s %s:%s %s" % \
        (EXCLUDE_DIRS, SERVER_IP, SERVER_DIR, LOCAL_DIR)
    print(rsync_cmd); os.system(rsync_cmd)
