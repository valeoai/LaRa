import torch.optim as optim

class LinearWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup: int):
        """
        Args:
            warmup: Number of warmup steps. Usually between 50
        """

        self.warmup = warmup
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            return epoch / self.warmup
        return 1.0