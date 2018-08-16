from my_modules.modules import trainer


class Trainer(trainer.Trainer):
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging."""

    def compute_loss(self, outputs, data):
        pass

    def setup_handler(self, epoch, mode, num_batch, ds_size):
        pass
