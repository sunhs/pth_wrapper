from pth_wrapper import trainer


class Trainer(trainer.Trainer):
    def compute_loss(self, outputs, labels):
        pass

    def create_handler(self, mode, num_batch):
        pass
