#!/usr/bin/env python
import os
import sys
import time

SERVER_IP = "biggpu"
EXPERIMENT_NAME = "*"
SERVER_DIR = "~/Desktop/MetaGPT/experiments/results"
LOCAL_DIR = SERVER_DIR

if __name__ == "__main__":
    print("Usage: ./download.py [exp_name]")
    print("Usage: ./download.py [exp_name] [server_ip]")
    time.sleep(1)

    if len(sys.argv) == 2:
        EXPERIMENT_NAME = sys.argv[1]
    elif len(sys.argv) == 3:
        EXPERIMENT_NAME = sys.argv[1]
        SERVER_IP = sys.argv[2]

    os.system("rsync -Phavz --stats %s:%s/%s %s" % \
        (SERVER_IP, SERVER_DIR, EXPERIMENT_NAME, LOCAL_DIR))
