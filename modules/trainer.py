import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataloader, sampler

from my_modules.modules import utils


class Trainer:
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging.

    Interfaces:
        compute_loss(self, outputs, labels)
        setup_handler(self, epoch, mode, num_batch, ds_size)

    Methods:
        setup_optimizer(self)
            default: Adam

        setup_lr_scheduler(self)
            defalut: None

        setup_dataloader(self, mode)
            default: pytorch dataloader

        get_input_size(self, data)
            defalut: For torch.Tensor, return its shape. For a list, recursively
            check if the first element is torch.Tensor and return its shape.
            Otherwise return None.

        format_data(self, data, mode)
            default: Construct Variables for the data.

        format_output(self, outputs, input_size, mode)
            default: Directly return the outputs. 
    
        cleanup_batch(self, handler, data, outputs, loss, batch, hz)
            default: Use handler.cleanup_batch(data, outputs, loss, batch, hz)

        cleanup_epoch(self, handler)
            default: Use handler.cleanup_epoch()
    """

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
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
                self.test_epoch()

    def train_epoch(self):
        epoch = self.latest_state + 1
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}\n'.format(utils.parse_time(t - s)))
        self.model.cpu()

        if (not getattr(self.config, 'SAVE_EPOCH_FREQ', None)
                or epoch % self.config.SAVE_EPOCH_FREQ == 0):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.config.STATE_DIR, '{}_{}.pth'.format(
                        self.config.STATE_PREFIX, epoch)))
        self.latest_state = epoch

    def test_epoch(self):
        epoch = self.latest_state
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n\n\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        print('\033[1;32m{}: epoch {:3d}/{:3d}\033[0m'.format(
            mode, epoch, self.config.MAX_EPOCHS))

        self.model.cuda(self.config.DEFAULT_GPU)

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
                output_device=self.config.DEFAULT_GPU)
        else:
            model = self.model

        data_loader = self.setup_dataloader(mode)
        num_batch = len(data_loader)
        handler = self.setup_handler(epoch, mode, num_batch,
                                     len(self.dataset[mode]))

        for i, data in enumerate(data_loader):
            model.zero_grad()
            s = time.time()
            input_size = self.get_input_size(data)
            data = self.format_data(data, mode)
            outputs = self.format_output(model(data), input_size, mode)
            loss = self.compute_loss(outputs, data)
            t = time.time()
            hz = outputs.size(0) / (t - s)

            self.cleanup_batch(handler, data, outputs, loss, i, hz)

            if mode == 'train':
                loss.backward()
                self.optimizer.step()

        self.cleanup_epoch(handler)

    def setup_optimizer(self):
        param_groups = utils.get_param_groups(self.model, self.config)
        optimizer = optim.Adam(param_groups)
        return optimizer

    def setup_lr_scheduler(self):
        return None

    def setup_dataloader(self, mode):
        dataset = self.dataset[mode]
        data_loader = dataloader.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE[mode],
            sampler=sampler.RandomSampler(dataset),
            num_workers=self.config.NUM_WORKERS,
            collate_fn=dataset.collate_fn)
        return data_loader

    def get_input_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.size()
        if isinstance(data, (list, tuple)):
            return self.get_input_size(data[0])
        return None

    def format_data(self, data, mode):
        inputs, labels = data
        return inputs, labels

    def format_output(self, outputs, input_size, mode):
        return outputs

    def cleanup_batch(self, handler, data, outputs, loss, batch, hz):
        handler.cleanup_batch(data, outputs, loss, batch, hz)

    def cleanup_epoch(self, handler):
        handler.cleanup_epoch()

    def compute_loss(self, outputs, data):
        raise NotImplementedError

    def setup_handler(self, epoch, mode, num_batch, ds_size):
        """A handler should have the following methods.
        cleanup_batch(self, outputs, labels, loss, batch, hz)
        cleanup_epoch()
        """
        raise NotImplementedError
