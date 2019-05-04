import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader, sampler

from . import utils


class Trainer:
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging.

    Interfaces:
        compute_loss(self, outputs, labels)
        create_handler(self, mode, num_batch)

    """

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config

        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_lr_scheduler()
        self.latest_state = utils.init_model(
            self.model, self.config.STATE_DIR, self.config.STATE_PREFIX,
            self.config.STATE_INDEX, self.config.PRETRAIN_PATH
        )

        if config.DEFAULT_GPU is not None:
            assert config.DEFAULT_GPU >= 0
            assert torch.cuda.is_available()
            self.device = torch.device('cuda:' + str(config.DEFAULT_GPU))
        else:
            self.device = torch.device('cpu')

    def train(self, test=True):
        """The training process."""
        for _ in range(self.latest_state + 1, self.config.MAX_EPOCHS):
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch=self.latest_state + 1)
            self.train_epoch()
            if test:
                self.test_epoch()

    def train_epoch(self):
        epoch = self.latest_state + 1
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}\n'.format(utils.parse_time(t - s)))

        if (
            not self.config.SAVE_EPOCH_FREQ or
            epoch % self.config.SAVE_EPOCH_FREQ == 0 or
            epoch == self.config.MAX_EPOCHS - 1
        ):
            self.model.cpu()
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.config.STATE_DIR,
                    '{}.pth-{:04d}'.format(self.config.STATE_PREFIX, epoch)
                )
            )
        self.latest_state = epoch

    def test_epoch(self):
        epoch = self.latest_state
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n\n\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        color_code = ''
        if sys.platform != 'win32':
            color_code = '\033[1;31m' if mode == 'train' else '\033[1;34m'
        end_color_code = '\033[0m' if sys.platform != 'win32' else ''
        print(
            color_code + \
            '{}: epoch {:3d}/{:3d}'.format(mode, epoch, self.config.MAX_EPOCHS) + \
            end_color_code
        )

        self.model.to(self.device)

        if mode == 'train':
            self.model.train()
            torch.set_grad_enabled(True)
        elif mode == 'test':
            self.model.eval()
            torch.set_grad_enabled(False)

        if len(self.config.GPUS) > 1:
            model = nn.DataParallel(
                self.model,
                self.config.GPUS,
                output_device=self.config.DEFAULT_GPU
            )
        else:
            model = self.model

        loader = self.create_dataloader(mode)
        handler = self.create_handler(mode, num_batch=len(loader))

        for i, data in enumerate(loader):
            if isinstance(data, tuple):
                data = list(data)
            model.zero_grad()
            tick = time.time()
            inputs, labels = self.parse_data(data, mode)
            outputs = self.parse_output(model(*inputs), mode)
            losses = self.compute_loss(outputs, labels)
            if mode == 'train':
                self.backward(losses)
                self.optimizer.step()

            handler.cleanup_batch(data, outputs, losses, i, tick)
        handler.cleanup_epoch()

    def create_optimizer(self):
        """Create an optimizer. Use the `get_param_groups` defined in
        `pth_wrapper.utils` to appaly different optimization schemes
        to different parameter sets.
        Default: Adam optimizer with the `param_groups` set up in your config
        file.

        Returns
        -------
        torch.optim.Optimizer
        """
        param_groups = utils.get_param_groups(self.model, self.config)
        optimizer = optim.Adam(param_groups)
        return optimizer

    def create_lr_scheduler(self):
        """Create a learning rate scheduler.
        Default: Use no scheduler.

        Returns
        -------
        torch.optim.lr_scheduler._LRScheduler
        """
        return None

    def create_dataloader(self, mode):
        """Set up the dataloader for the current mode for each epoch.

        Parameters
        ----------
        mode: str
            `train` or `test`.

        Returns
        -------
        torch.utils.data.DataLoader
        """
        dataset = self.dataset[mode]
        shuffle = True if mode is 'train' else False
        loader = dataloader.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE[mode],
            shuffle=shuffle,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=dataset.collate_fn
        )
        return loader

    def parse_data(self, data, mode):
        """Separate inputs and labels. Move to specified devices and preprocess
        them.

        Parameters
        ----------
        data: list of torch.Tensor
            [input_1[, input_2, ...], label_1[, label_2, ...]]
        mode: str
            `train` or `test`.

        Returns
        -------
        tuple
            The first element is the list of inputs,
            the second the list of labels.
        """
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                data[i] = data[i].to(self.device)
        return data[:-1], data[-1:]

    def parse_output(self, outputs, mode):
        """Make the outputs suitable for computing loss.

        Parameters
        ----------
        outputs: torch.Tensor or list of torch.Tensor
        mode: str
            `train` or `test`.

        Returns
        -------
        list of torch.Tensor
        """
        return [outputs] if isinstance(outputs, torch.Tensor) else outputs

    def backward(self, losses):
        for loss in losses:
            loss.backward()

    def compute_loss(self, outputs, labels):
        """Compute the loss.

        Parameters
        ----------
        outputs: list of torch.Tensor
        labels: list of torch.Tensor

        Returns
        -------
        list:
            Contain different losses. Even if there's only 1 loss, should wrap
            it in a list.
        """
        raise NotImplementedError

    def create_handler(self, mode, num_batch):
        """Sets up a handler to perform logging or postprocessing after each
        batch and each epoch. A handler should have the following methods:
            `cleanup_batch(self, data, outputs, losses, batch, tick)`
            `cleanup_epoch(self)`
        Specifically, `batch` is the current batch index and `tick` is the start
        time for processing the current batch.

        Parameters
        ----------
        mode: str
            `train` or `test`.
        num_batch: int
            Number of batches.

        Returns
        -------
        Object
            The handler.
        """
        raise NotImplementedError
