from util import *
import os
import torch
from abc import ABC, abstractmethod
import torch.nn as nn
import util.metrics as metrics
from schedulers import get_scheduler

class BaseModel(object):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.wait_over = False
        self.start_forward = True
        self.wait_epoch = 0

        self.gpu_ids = opt.gpu_ids
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.o_save_dir = self.save_dir

        if self.opt.visualize and self.opt.vis_method is not None and self.opt.l_state != 'train':
            if self.opt.threshold is None:
                threshold = 0.5
            else:
                threshold = self.opt.threshold
            self.vis_dir = os.path.join(opt.vis_dir, opt.name+'({})'.format(repr(self.opt.vis_layer_names)))
            mkdir(self.vis_dir)

        self.loss_names = ['c'] # used to update networks,
        self.s_metric_names = ['accuracy'] # scalar metric, stat local infomation
        self.g_metric_names = [] # scalar metric, stat global infomation
        self.t_metric_names = ['cmatrix'] # table or matrix metric
        self.buffer_names = []

        self.net_names = []
        self.optimizers = []

        self.valid_metric = 'accuracy'
        self.scheduler_metric = 'accuracy'

        self.best_m_value = -1
        self.c_grad_iter = 0

        self.buffer_ginput_ids = []
        self.buffer_gscores = []
        self.buffer_glabels = []
        self.buffer_gpreds = []

    @staticmethod
    def modify_commandline_options(parser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """
        return parser

    def gen_meters(self):
        name_types = ['loss','s_metric','t_metric','g_metric']
        meters = {}
        for ntype in name_types:
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                meters[name] = metrics.Meter()
        return meters

    def update_metrics(self, m_type = 'local'):
        if not self.start_forward and m_type != 'global':
            return

        if m_type == 'global':
            name_types = ['t_metric', 'g_metric']
        else:
            name_types = ['loss', 's_metric']

        for ntype in name_types:
            cal_func = getattr(self,'cal_' + ntype)
            cal_func()
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                self.update_meters(ntype, name)

    def update_meters(self, ntype, name):
        value = getattr(self, ntype + '_' + name)

        if isinstance(value,torch.Tensor):
            value = value.detach().cpu().numpy()

        if isinstance(value, np.ndarray) and ntype != 't_metric':
            value = value.item()

        if ntype != 't_metric':
            self.meters[name].update(value,self.input_size)
        else:
            self.meters[name].update(value,1)

    def reset_meters(self):
        name_types = ['loss', 's_metric', 't_metric', 'g_metric']

        for ntype in name_types:
            name_list = getattr(self, ntype + '_names')
            for name in name_list:
                value = getattr(self, ntype + '_' + name)
                self.meters[name].reset()

    @abstractmethod
    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    def get_parameters(self):
        names = self.net_names
        p_list = []
        for name in names:
            net = getattr(self, 'net_' + name)
            p_list.append(net.parameters())

        if len(p_list) == 1:
            return p_list[0]
        else:
            n_p_list = []
            for p in p_list:
                tmp_p = {}
                tmp_p['params'] = p
                n_p_list.append(tmp_p)
            return n_p_list

    def clear_info(self):
        for name in self.buffer_names:
            if name == 'names':
                continue
            tmp_buffer = getattr(self,'buffer_' + name)
            if len(tmp_buffer) > 0:
                if isinstance(tmp_buffer[0],list):
                    tmp_buffer = [[] for _ in range(len(tmp_buffer))]
                else:
                    tmp_buffer = []
            setattr(self,'buffer_' + name,tmp_buffer)
            value = getattr(self, 'buffer_' + name)

    def set_optimizer(self,opt):
        if opt.op_name == 'SGD':
            self.optimizer = torch.optim.SGD(self.get_parameters(), lr=opt.lr, momentum = opt.momentum, nesterov=opt.nesterov,
                                             weight_decay = opt.weight_decay)
        elif opt.op_name == 'Adam':
            self.optimizer = torch.optim.Adam(self.get_parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.weight_decay)
        elif opt.op_name == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.get_parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay = opt.weight_decay)
        self.optimizers = [self.optimizer]

    @abstractmethod
    def forward(self):
        pass

    def validate_parameters(self):
        self.start_forward = True

        if self.opt.vis_method in ['gradcam']:
            self.forward()
        else:
            with torch.no_grad():
                self.forward()
        self.stat_info()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.start_forward = True
        self.forward()  # first call forward to calculate intermediate results
        self.backward()  # calculate gradients for network G
        self.stat_info()
        self.c_grad_iter += 1

        if self.c_grad_iter == self.opt.grad_iter_size:
            self.optimizer.step()  # update gradients for network G
            self.optimizer.zero_grad()  # clear network G's existing gradients
            self.c_grad_iter = 0

    def stat_info(self):
        self.label = self.label.cpu().long()
        self.y = self.y.cpu()
        self.score = self.score.cpu()

        if self.y.shape[1] > 1:
            pred = torch.argmax(self.y, dim=1)
        else:
            pred = (self.y.view(-1) > self.opt.recall_thred).long()

        self.pred = pred.expand_as(self.label)

        self.buffer_gscores.extend(self.score.view(-1).tolist())
        self.buffer_glabels.extend(self.label.view(-1).tolist())
        self.buffer_gpreds.extend(self.pred.view(-1).tolist())
        self.buffer_ginput_ids.extend(self.input_id)

    def get_buffer_names(self):
        v_names = list(self.__dict__.keys())
        b_names = [v.replace('buffer_','') for v in v_names if v.startswith('buffer')]
        return b_names

    def zero_grad(self):
        for n_name in self.net_names:
            net = getattr(self,'net_' + n_name)
            net.zero_grad()

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.buffer_names = self.get_buffer_names()
        self.meters = self.gen_meters()
        self.schedulers = []

        if opt.load_dir is not None:
            load_suffix = 'optimal'
            self.load_networks(load_suffix, load_dir=opt.load_dir)

        if opt.l_state == 'train':
            self.set_optimizer(opt)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        else:
            load_suffix = 'optimal'
            self.load_networks(load_suffix)
            self.print_networks(opt.verbose)

        # visualization method
        if opt.visualize and opt.vis_method is not None:
            self.vis_method = vis_method_dict[opt.vis_method](self)

        for name in self.net_names:
            net = getattr(self,'net_' + name)
            net.cuda(self.gpu_ids[0])
            setattr(self,'net_' + name,nn.DataParallel(net,opt.gpu_ids))

            if self.opt.l_state == 'train':
                net.train()
            else:
                net.eval()

    def set_l_state(self, l_state):
        assert l_state in ['train', 'valid', 'test']
        self.opt.l_state = l_state

    def cal_loss(self):
        loss_list = []
        self.loss_c = nn.CrossEntropyLoss()(self.y, self.label)
        loss_list.append(self.loss_c)
        return loss_list

    def cal_s_metric(self):
        if 'accuracy' in self.s_metric_names:
            self.s_metric_accuracy = metrics.accuracy(self.y, self.label)[0]

    def cal_g_metric(self):
        if 'precision' in self.g_metric_names:
            self.g_metric_precision = metrics.precision(self.t_metric_cmatrix, 1)
        if 'recall' in self.g_metric_names:
            self.g_metric_recall = metrics.recall(self.t_metric_cmatrix, 1)
        if 'fscore' in self.g_metric_names:
            self.g_metric_fscore = metrics.f_score(self.t_metric_cmatrix, 1)
        if 'auc' in self.g_metric_names:
            self.g_metric_auc = metrics.auc_score(self.buffer_glabels, self.buffer_gscores)

    def cal_t_metric(self):
        if 'cmatrix' in self.t_metric_names:
            self.t_metric_cmatrix = metrics.comfusion_matrix(self.buffer_gpreds, self.buffer_glabels,self.opt.num_classes)

    def backward(self):
        self.update_metrics('local')
        for name in self.loss_names:
            loss = getattr(self,'loss_' + name) / self.opt.grad_iter_size
            loss.backward()

    def validation(self,dataset,visualizer,valid_iter):
        self.eval()
        if self.opt.l_state != 'test':
            self.set_l_state('valid')
        iter_time_meter = metrics.TimeMeter()
        data_time_meter = metrics.TimeMeter()

        data_time_meter.start()
        iter_time_meter.start()

        for i, data in enumerate(dataset):  # inner loop within one epoch
            data_time_meter.record(n = self.opt.batch_size)
            iter_time_meter.start()
            self.set_input(data)
            self.validate_parameters()
            self.update_metrics('local')

            iter_time_meter.record()

            if i % self.opt.v_print_freq == 0: 
                visualizer.print_current_info(-1, i, self, iter_time_meter.val, data_time_meter.val)

            data_time_meter.start()
            iter_time_meter.start()

        self.update_metrics('global')
        visualizer.print_global_info(-1, -1, self, iter_time_meter.sum/60, data_time_meter.sum/60)
        self.save_stat_info()
        self.reset_meters()
        self.clear_info()
        self.train()
        self.set_l_state('train')

    def plot_special_info(self):
        pass

    def print_special_info(self,log_name):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.eval()

    def train(self):
        for name in self.net_names:
            net = getattr(self, 'net_' + name)
            net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        if self.opt.vis_method in ['gradcam']:
            self.forward()
            self.compute_visuals()
        else:
            with torch.no_grad():
                self.forward()
                self.compute_visuals()

    def get_metric(self,metric_name):
        try:
            value = float(getattr(self, 's_metric_' + metric_name))
        except:
            value = float(getattr(self, 'g_metric_' + metric_name))
        return value

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def save_networks(self, epoch, visualizer):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        print('valid metric ' + str(self.valid_metric))
        tmp_v_value = self.get_metric(self.valid_metric)
        print('v value ' + str(tmp_v_value))
        if tmp_v_value > self.best_m_value:
            self.best_m_value = tmp_v_value
            old_save_dir_sh = self.save_dir.replace('(','\(').replace(')','\)')
            self.save_dir = self.o_save_dir + '({}={:.3f})'.format(self.valid_metric, self.best_m_value)
            new_save_dir_sh = self.save_dir.replace('(','\(').replace(')','\)')
            os.system('mv ' + old_save_dir_sh + ' ' + new_save_dir_sh)

            pred_fname = osp.join(new_save_dir_sh, 'pred_result.txt')
            if osp.exists(pred_fname):
                n_pred_fname = pred_fname.replace('pred','optimal_pred')
                os.system('mv ' + pred_fname + ' ' + n_pred_fname)

            log_parts = visualizer.log_name.split('/')
            log_parts[-2] = visualizer.o_log_p2 + '({}={:.3f})'.format(self.valid_metric, self.best_m_value)
            visualizer.log_name = ''
            for p in log_parts:
                visualizer.log_name += p + '/'
            visualizer.log_name = visualizer.log_name[:-1]

            self.wait_epoch = 0

            for name in self.net_names:
                if isinstance(name, str):
                    save_filename = '%s_net_%s.pth' % (epoch, name)
                    save_path = os.path.join(self.save_dir, save_filename)
                    net = getattr(self, 'net_' + name)

                    if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                        torch.save(net.module.state_dict(), save_path)
                    else:
                        torch.save(net.state_dict(), save_path)
        else:
            self.wait_epoch += 1
            if self.wait_epoch > self.opt.patient_epoch:
                self.wait_over = True

    def load_networks(self, epoch, load_dir = None):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.net_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                if load_dir is None:
                    load_dir = self.save_dir
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net_' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location = 'cpu')
                net.load_state_dict(state_dict)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.net_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    @abstractmethod
    def save_stat_info(self):
        pass

    def next_epoch(self):
        pass

    def get_metric_kind(self, m_name):
        if m_name in self.loss_names:
            return 'loss'
        elif m_name in self.s_metric_names:
            return 's_metric'
        elif m_name in self.t_metric_names:
            return 't_metric'
        elif m_name in self.g_metric_names:
            return 'g_metric'

        AssertionError(False, 'This metric is not in this model')
