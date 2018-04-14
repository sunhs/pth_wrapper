import datetime
import math
import os
import time

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from my_modules import utils


class Trainer:
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging."""

    def __init__(self, model, ds, config):
        self.model = model
        self.ds = ds
        self.config = config
        self.latest_state = -1

        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_lr_scheduler()
        self.latest_state = utils.load_state_dict(
            self.model, self.config.PRETRAIN_PATH, self.config.STATE_DIR,
            self.config.STATE_PREFIX)

    def train(self, test=True):
        """The training process."""
        for _ in range(self.latest_state + 1, self.config.MAX_EPOCHS):
            if self.lr_scheduler:
                self.lr_scheduler.step(epoch=self.latest_state + 1)
            self.train_epoch()
            if test:
                self.eval_epoch()

    def train_epoch(self):
        epoch = self.latest_state + 1
        self.ds.shuffle_train_set()
        print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}'.format(utils.parse_time(t - s)))
        self.model.cpu()
        torch.save(self.model.state_dict(), os.path.join(
            self.config.STATE_DIR, '{}_{}.pth'.format(
                self.config.STATE_PREFIX, epoch)))
        self.latest_state = epoch

    def eval_epoch(self):
        epoch = self.latest_state
        print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n\n\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        self.model.cuda(self.config.DEFAULT_GPU)

        if mode == 'train':
            self.model.train()
        elif mode == 'test':
            self.model.eval()

        if len(self.config.GPUS) > 1:
            model = nn.DataParallel(
                self.model, self.config.GPUS,
                output_device=self.config.DEFAULT_GPU)
        else:
            model = self.model

        batch_size = self.config.BATCH_SIZE[mode]
        num_batch = int(math.ceil(self.ds.ds_size[mode] / batch_size))
        logger = self.setup_logger(
            epoch, mode, num_batch, self.ds.ds_size[mode])

        for i in range(num_batch):
            model.zero_grad()
            s = time.time()
            inputs, labels = self.ds.next_batch(batch_size, mode)
            input_size = self.get_input_size(inputs)
            inputs, labels = self.format_data(inputs, labels, mode)
            outputs = self.format_output(model(inputs), input_size, mode)
            loss = self.compute_loss(outputs, labels)
            t = time.time()
            hz = outputs.size(0) / (t - s)

            self.cleanup_batch(logger, outputs, labels, loss, i, hz)

            if mode == 'train':
                loss.backward()
                # import ipdb
                # ipdb.set_trace()
                self.optimizer.step()

        self.cleanup_epoch(logger)

    def setup_optimizer(self):
        param_groups = utils.get_param_groups(self.model, self.config)
        optimizer = optim.Adam(param_groups)
        return optimizer

    def setup_lr_scheduler(self):
        return None

    def get_input_size(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return inputs.size()
        elif isinstance(inputs, list):
            return self.get_input_size(inputs[0])
        else:
            return None

    def format_data(self, inputs, labels, mode):
        inputs = Variable(inputs.cuda(self.config.DEFAULT_GPU))
        labels = Variable(labels.cuda(self.config.DEFAULT_GPU))

        if mode == 'test':
            inputs.volatile = True

        return inputs, labels

    def format_output(self, outputs, input_size, mode):
        return outputs

    def cleanup_batch(self, logger, outputs, labels, loss, batch, hz):
        logger.cleanup_batch(outputs, labels, loss, batch, hz)

    def cleanup_epoch(self, logger):
        logger.cleanup_epoch()

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def setup_logger(self, epoch, mode, num_batch, ds_size):
        raise NotImplementedError
