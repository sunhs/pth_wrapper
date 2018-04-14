import argparse
import importlib
import os
import shutil
import sys


sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=int, default=1)
    args = parser.parse_args()
    config = importlib.import_module('myconfs.config_{}'.format(args.c))
    shutil.rmtree(config.MODEL_DIR)
