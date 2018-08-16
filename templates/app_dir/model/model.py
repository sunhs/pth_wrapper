import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        """
        Initializes the model instance and calls the `initialize` method.

        :param config: The global config.
        """
        super(Model, self).__init__()
        self.initialize()

    def initialize(self):
        """Initializes model parameters.

        :returns: None
        :rtype: None
        """
        pass

    def weights_init(self, m):
        """An example function to initialize model parameters. This method
        is not necessary since you can do the job directly in `initialize`.

        :param m: torch.nn.Module
        :returns: None
        :rtype: None

        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 1e-4)

    def forward(self, inputs):
        pass
