import argparse
import importlib
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cont(checkpoint, config):
    mod_log_file(checkpoint, config)
    rm_state_file(checkpoint, config)
    mod_train_log_file(checkpoint, config)


def rm_state_file(checkpoint, config):
    checkpoint_fname = '{}_{}.pth'.format(config.STATE_PREFIX, checkpoint)
    file_list = os.listdir(config.STATE_DIR)
    for file_name in file_list:
        if file_name != checkpoint_fname:
            os.remove(os.path.join(config.STATE_DIR, file_name))


def mod_log_file(checkpoint, config):
    with open(os.path.join(config.MODEL_DIR, 'log.pickle'), 'rb') as f:
        log = pickle.load(f)
    for mode in log:
        for metric in log[mode]:
            if len(log[mode][metric]) > checkpoint:
                log[mode][metric] = log[mode][metric][:checkpoint + 1]
    with open(os.path.join(config.MODEL_DIR, 'log.pickle'), 'wb') as f:
        pickle.dump(log, f)


def mod_train_log_file(checkpoint, config):
    with open(os.path.join(config.MODEL_DIR, 'train_log.pickle'), 'rb') as f:
        train_log = pickle.load(f)
    if len(train_log) > checkpoint:
        train_log = train_log[:checkpoint + 1]
    with open(os.path.join(config.MODEL_DIR, 'train_log.pickle'), 'wb') as f:
        pickle.dump(train_log, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', type=str)
    parser.add_argument('-c', '--config', type=int, default=1)
    parser.add_argument(
        'checkpoint', type=int, help='Checkpoint to start from.'
    )
    args = parser.parse_args()
    config = importlib.import_module(
        '{}.confs.config_{}'.format(args.app, args.config)
    )
    checkpoint = args.checkpoint
    cont(checkpoint, config)


if __name__ == '__main__':
    main()
