import argparse
import datetime
import importlib

import torch


def main():
    print(datetime.datetime.now())

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--app', type=str, help='The app to use.')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=['all', 'train', 'test'],
        default='all',
        help='The mode to run. `all` will train and test epochs alternately. \
        `train` and `test` will train and test 1 epoch, respectively.'
    )
    parser.add_argument(
        '-c', '--config', type=int, default=1, help='Which config file to use.'
    )
    args = parser.parse_args()

    config = importlib.import_module(
        '{}.confs.config_{}'.format(args.app, args.config)
    )
    torch.cuda.set_device(config.DEFAULT_GPU)

    print('==========>> build dataset')
    dataset_module = importlib.import_module(
        '{}.dataset.dataset'.format(args.app)
    )
    dataset = {
        'train': dataset_module.Dataset(config, 'train'),
        'test': dataset_module.Dataset(config, 'test')
    }

    print('==========>> build model')
    model_module = importlib.import_module('{}.model.model'.format(args.app))
    model = model_module.Model(config)

    print('==========>> build trainer')
    trainer_module = importlib.import_module(
        '{}.trainer.trainer'.format(args.app)
    )
    trainer = trainer_module.Trainer(model, dataset, config)

    print('==========>> start to run model')
    if args.mode == 'all':
        trainer.train(test=False)
    elif args.mode == 'train':
        trainer.train_epoch()
    elif args.mode == 'test':
        trainer.test_epoch()


if __name__ == '__main__':
    main()
