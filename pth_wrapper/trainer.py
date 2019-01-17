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
        setup_handler(self, epoch, mode, num_batch, ds_size)

    """

    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.latest_state = -1

        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_lr_scheduler()
        self.latest_state = utils.load_state_dict(
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
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'train')
        t = time.time()
        print('train 1 epoch in {}\n'.format(utils.parse_time(t - s)))
        self.model.cpu()

        if (
            not self.config.SAVE_EPOCH_FREQ or
            epoch % self.config.SAVE_EPOCH_FREQ == 0 or
            epoch == self.config.MAX_EPOCHS
        ):
            torch.save(
                self.model.state_dict(),
                os.path.join(
                    self.config.STATE_DIR,
                    '{}_{}.pth'.format(self.config.STATE_PREFIX, epoch)
                )
            )
        self.latest_state = epoch

    def test_epoch(self):
        epoch = self.latest_state
        # print(datetime.datetime.now())
        s = time.time()
        self._process_epoch(epoch, 'test')
        t = time.time()
        print('test 1 epoch in {}\n\n\n'.format(utils.parse_time(t - s)))

    def _process_epoch(self, epoch, mode):
        color_code = '\033[1;32m' if sys.platform != 'win32' else ''
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

        data_loader = self.setup_dataloader(mode)
        handler = self.setup_handler(
            epoch,
            mode,
            num_batch=len(data_loader),
            ds_size=len(self.dataset[mode])
        )

        for i, data in enumerate(data_loader):
            loss = None
            model.zero_grad()
            s = time.time()
            data, real_batch_size = self.format_data(data, mode)
            inputs = self.get_input(data, mode)
            outputs = self.format_output(model(inputs), data, mode)
            if mode == 'train':
                loss = self.compute_loss(outputs, data)
                self.optimize(loss)

            t = time.time()
            hz = real_batch_size / (t - s)

            self.cleanup_batch(handler, data, outputs, loss, i, hz)
        self.cleanup_epoch(handler)

    def setup_optimizer(self):
        """Set up an optimizer. Use the `get_param_groups` defined in
        `my_modules.modules.utils` to appaly different optimization schemes
        to different parameter sets.
        Default: Adam optimizer with the `param_groups` set up in your config
        file.

        :returns: The optimizer.
        :rtype: torch.optim.Optimizer

        """
        param_groups = utils.get_param_groups(self.model, self.config)
        optimizer = optim.Adam(param_groups)
        return optimizer

    def setup_lr_scheduler(self):
        """Set up a learning rate scheduler.
        Default: Use no scheduler.

        :returns: The lr scheduler.
        :rtype: torch.optim.lr_scheduler._LRScheduler

        """
        return None

    def setup_dataloader(self, mode):
        """Set up the dataloader for the current mode for each epoch.
        Default: A default pytorch dataloader with a user-defined dataset,
        batch_size, num_workers, a random sampler and a collate function
        defined in the dataset.

        :param mode: The current mode, either 'train' or 'test'.
        :returns: A pytorch dataloader.

        """
        dataset = self.dataset[mode]
        data_loader = dataloader.DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE[mode],
            sampler=sampler.RandomSampler(dataset),
            num_workers=self.config.NUM_WORKERS,
            collate_fn=dataset.collate_fn
        )
        return data_loader

    def format_data(self, data, mode):
        """Do some data processing before fed to the model. Returns the
        processed data and a real batch size.
        Default: If the data is not a list/tuple, move it to the cuda device
        specified in the config file and compute the real batch size according
        to `data`. Otherwise do the same to each element and compute the real
        batch size according to the first element of `data`.

        :param data: The data generated by the dataloader.
        :param mode: The current mode, either 'train' or 'test'.
        :returns: The formated data and a real batch size. If `data` is a
            list/tuple, still wrap the formatted data in a list/tuple.
        :rtype: Tuple

        """
        if isinstance(data, (list, tuple)):
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            return data, data[0].size(0)
        return data.to(self.device), data.size(0)

    def get_input(self, data, mode):
        """Produce the input data which will be directly fed to the model.
        Default: Directly returns `data` if it's not a list/tuple. Otherwise
        returns the first element.

        :param data: The formatted data.
        :param mode: The current mode, either 'train' or 'test'.
        :returns: The ready input data.
        :rtype: torch.Tensor

        """
        if isinstance(data, (list, tuple)):
            return data[0]
        return data

    def format_output(self, outputs, data, mode):
        """Process the model output according to your rule. It may depend on
        the data and the current mode.
        Default: Directly return the output.

        :param outputs: The model output.
        :param data: The data generated by the dataloader.
        :param mode: The current mode, either 'train' or 'test'.
        :returns: The formatted output.

        """
        return outputs

    def optimize(self, loss):
        """Backward the loss and optimize the parameters.
        Default: If `loss` is a list/tuple, call `backward` on each of the
        elements. Otherwise call it on `loss`. At the end, call `step` of the
        optimizer.

        """
        if isinstance(loss, (list, tuple)):
            for e in loss:
                e.backward()
        else:
            loss.backward()
        self.optimizer.step()

    def cleanup_batch(self, handler, data, outputs, loss, batch, hz):
        handler.cleanup_batch(data, outputs, loss, batch, hz)

    def cleanup_epoch(self, handler):
        handler.cleanup_epoch()

    def compute_loss(self, outputs, data):
        """Computes one or more losses according to the model output and the
        data generated by the dataloader. If multiple losses exist, you can
        sum them up into one loss. You can also just leave them as they are
        and return them, since in the `optimize` method you'll have a chance
        to perform the backward job on all of them. And in this way you can
        log multiple losses in the handler for better training monitoring.

        :param outputs: The model outputs.
        :param data: The data generated by the dataloader and formatted by
            `format_data`.
        :returns: The loss / losses.

        """
        raise NotImplementedError

    def setup_handler(self, epoch, mode, num_batch, ds_size):
        """Sets up a handler to perform logging or postprocessing after each
        batch and each epoch. A handler should have the following methods:
            `cleanup_batch(self, data, outputs, loss, batch, hz)`
            `cleanup_epoch(self)`
        Specifically, `batch` is the current batch index and `hz` is the speed
        for processing the current batch.

        :param epoch: The current epoch index.
        :param mode: The current mode, either `train` or `test`.
        :param num_batch: Number of batches.
        :param ds_size: The size of the dataset.
        :returns: The handler.

        """
        raise NotImplementedError
