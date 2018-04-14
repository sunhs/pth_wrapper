from my_modules.modules import trainer


class ProjectNameTrainer(trainer.Trainer):
    """The model trainer.
    It takes over the task of building computation graph, loading parameters,
    training, testing and logging."""

    def compute_loss(self, outputs, labels):
        raise NotImplementedError

    def setup_logger(self, epoch, mode, num_batch, ds_size):
        raise NotImplementedError
