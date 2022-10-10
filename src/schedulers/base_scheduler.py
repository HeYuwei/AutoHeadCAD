from torch.optim.lr_scheduler import _LRScheduler

class BaseScheduler(_LRScheduler):
    def __init__(self, optimizer, opt):
        # super(BaseScheduler, self).__init__(optimizer)
        self.wait_epoch = 0
        self.last_epoch = 0
        self.lr = opt.lr
        self.opt = opt
        self.warm_epoch = opt.warm_epoch
        self.warm_start_lr = self.lr * opt.warm_start_factor

    def step(self, epoch=None):
        pass

    def warm_up(self, epoch):
        if epoch >= self.warm_epoch:
            return self.lr

        warm_epoch = self.opt.warm_epoch
        lr_bias = self.opt.lr - self.warm_start_lr
        lr = self.warm_start_lr + lr_bias * epoch / (warm_epoch - 1)
        return lr