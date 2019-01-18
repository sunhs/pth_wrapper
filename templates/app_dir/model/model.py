import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        """
        Initializes the model instance and calls the `initialize` method.

        :param config: The global config.
        """
        super(Model, self).__init__()

    def initialize(self):
        pass

    def forward(self, x):
        pass
