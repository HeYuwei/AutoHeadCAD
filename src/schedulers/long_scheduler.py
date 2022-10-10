from torch.optim.lr_scheduler import _LRScheduler

class longScheduler(_LRScheduler):
    def __init__(self, optimizer,opt):
        self.gamma = opt.gamma
        self.power = opt.power
        self.lr = opt.lr
        self.weight_decay = opt.weight_decay
        self.optimizer = optimizer
        super(longScheduler, self).__init__(optimizer)

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        lr = self.lr * (1 + self.gamma * epoch) ** (-self.power)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr * param_group['lr_mult']
            param_group['weight_decay'] = self.weight_decay * param_group['decay_mult']

    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--gamma', type=float, default=0.001, help='scheduler gamma')
        parser.add_argument('--power', type=float, default=0.75, help='scheduler power')
        return parser