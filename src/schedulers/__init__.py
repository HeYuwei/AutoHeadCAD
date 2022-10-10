import importlib
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from .base_scheduler import BaseScheduler


basic_schedulers = ['linear','step','plateau','cosine']
metric_schedulers = ['eco','plateau','mstep']

def get_option_setter(scheduler_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    if scheduler_name not in basic_schedulers:
        scheduler_class = find_scheduler_using_name(scheduler_name)
    return scheduler_class.modify_commandline_options


def find_scheduler_using_name(scheduler_name):
    """Import the module "schedulers/[scheduler_name]_scheduler.py".

    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    scheduler_filename = "schedulers." + scheduler_name + "_scheduler"
    schedulerlib = importlib.import_module(scheduler_filename)
    scheduler = None
    target_scheduler_name = scheduler_name.replace('_', '') + 'scheduler'
    for name, cls in schedulerlib.__dict__.items():
        if name.lower() == target_scheduler_name.lower() \
           and issubclass(cls, _LRScheduler):
            scheduler = cls

    if scheduler is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (scheduler_filename, target_scheduler_name))
        exit(0)

    return scheduler


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.T_max, eta_min=opt.eta_min)
    elif opt.lr_policy == 'cosineWR':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=opt.T_0, T_mult=opt.T_mult, eta_min=opt.eta_min)
    else:
        c_scheduler = find_scheduler_using_name(opt.lr_policy)
        scheduler = c_scheduler(optimizer,opt)
        # return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
