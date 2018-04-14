import argparse
import importlib

import project_name_dataset
import project_name_trainer


def load_model(config):
    model_module = importlib.import_module('models.{}'.format(
        config.MODEL_NAME))
    model = getattr(model_module, config.MODEL_NAME)(config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', type=str, choices=['all', 'train', 'test'], default='all')
    parser.add_argument('-c', type=int, default=1)
    args = parser.parse_args()

    config = importlib.import_module('myconfs.config_{}'.format(args.c))

    print('==========>> init dataset')
    dataset = {
        'train': project_name_dataset.ProjectNameDataset(config, 'train'),
        'test': project_name_dataset.ProjectNameDataset(config, 'test')
    }

    print('==========>> init model')
    model = load_model(config)

    print('==========>> init trainer')
    model_trainer = project_name_trainer.ProjectNameTrainer(
        model, dataset, config)

    print('==========>> start to run model')
    if args.m == 'all':
        model_trainer.train(test=True)
    elif args.m == 'train':
        model_trainer.train_epoch()
    elif args.m == 'test':
        model_trainer.test_epoch()


if __name__ == '__main__':
    main()
