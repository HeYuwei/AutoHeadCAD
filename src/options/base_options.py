import argparse
import os
from util import basic
import torch
import models
import datasets
import schedulers


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.parser = self.gather_options()
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, default='ct_experiment', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--load_dir', type=str, default=None, help='model paths to be loaded')
        parser.add_argument('--vis_dir', type=str, default='../vis', help='viualized data are saved here')

        # model parameters
        parser.add_argument('--model', type=str, default='testscan', help='chooses which model to use. [trainscan | testscan | ...]')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--num_classes', type=int,default=2, help='class number of data')
        parser.add_argument('--grad_iter_size', type=int, default=1, help='# Update networks every n iters, which is used to simulate large batch size')
        parser.add_argument('--l_state', type=str, default='valid', help='learning state')
        parser.add_argument('--recall_thred', type=float, default=0.5, help='recall thred')
        parser.add_argument('--vis_layer_names', type=list, default=['backbone.layer4'], help='the names of visible layers')
        parser.add_argument('--vis_method', type=str, default='gradcam')
        parser.add_argument('--visualize', type=int, default=1)
        parser.add_argument('--cam_postprocess', type=str, default='slicenormal_weight')
        parser.add_argument('--cam_thresh', type=float, default=0.8)

        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='scan', help='chooses train dataset')
        parser.add_argument('--dataset', type=object, default=None, help='training dataset object')
        parser.add_argument('--v_dataset_mode', type=str, default='vis', help='chooses valid dataset')
        parser.add_argument('--v_dataset', type=object, default=None, help='valid|test dataset object')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--v_batch_size', type=int, default=1, help='valid input batch size')
        parser.add_argument('--serial_batches',type=bool, default=True, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')

        #data augumentation
        parser.add_argument('--preprocess', type=str, default='resize', help='data augumintation [resize | crop | scale | \
                                                                                 translate | rotate | shear | elastc | flip | contrast | clane]')
        parser.add_argument('--load_size', type=int, default=512, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=512, help='then crop to this size')
        parser.add_argument('--scale_per_x', type=tuple, default=(0.7,1.3), help='the range of image to scale, horizontal')
        parser.add_argument('--scale_per_y', type=tuple, default=(0.7,1.3), help='the range of image to scale, vertical')
        parser.add_argument('--max_dataset_size', type=float,default=float('inf'),help='')
        parser.add_argument('--translate_pix_x', type=tuple, default=(-30, 30),
                            help='the pixcel range of image to translate, horizontal')
        parser.add_argument('--translate_pix_y', type=tuple, default=(-30, 30),
                            help='the pixcel range of image to translate, vertical')
        parser.add_argument('--rotate_der', type=tuple, default=(-20, 20), help='rotate range')
        parser.add_argument('--shear_der', type=tuple, default=(-20, 20), help='shear range')
        parser.add_argument('--elastic_alpha', type=tuple, default=(0,3), help='elastic_alpha range')
        parser.add_argument('--contrast_gain', type=tuple, default=(3, 10), help='contrast_gain')
        parser.add_argument('--contrast_cutoff', type=tuple, default=(0.4, 0.6), help='contrast_cutoff')
        parser.add_argument('--clane_limit', type=tuple, default=(1, 10), help='clane_limit')

        parser.add_argument('--flip_rate', type=float, default=0.5, help='flip_rate')
        parser.add_argument('--scale_rate', type=float, default=0.5, help='scale_rate')
        parser.add_argument('--translate_rate', type=float, default=0.5, help='translate_rate')
        parser.add_argument('--rotate_rate', type=float, default=0.5, help='rotate_rate')
        parser.add_argument('--shear_rate', type=float, default=0.5, help='shear_rate')
        parser.add_argument('--elastic_rate', type=float, default=0.5, help='elastic_rate')
        parser.add_argument('--contrast_rate', type=float, default=0.5, help='contrast_rate')
        parser.add_argument('--clane_rate', type=float, default=0.5, help='clane_rate')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--repeat_iter', type=int, default=0, help='train under the same setting for several times')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        
        #visualization parameters
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        parser.add_argument('--v_print_freq', type=int, default=50,
                            help='frequency of showing training results on console')
        parser.add_argument('--display_freq', type=int, default=400,
                            help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4,
                            help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_server', type=str, default="http://localhost",
                            help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main',
                            help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8099, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000,
                            help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=10,
                            help='frequency of showing training results on console')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,conflict_handler="resolve")
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = datasets.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        if opt.dataset_mode != opt.v_dataset_mode:
            dataset_name = opt.v_dataset_mode
            dataset_option_setter = datasets.get_option_setter(dataset_name)
            parser = dataset_option_setter(parser)

        # modify scheduler-related parser options
        # save and return the parser
        return parser

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # basic.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write(message)
        #     opt_file.write('\n')

    def opt_revise(self,opt):
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        # self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')

        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
                
        return opt

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.parser.parse_args()
        # opt.isTrain = self.isTrain   # train or test
        #
        self.opt = self.opt_revise(opt)
        return self.opt


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        # network saving and loading parameters
        parser.add_argument('--valid_model', type=bool, default=True, help='valid the model')
        parser.add_argument('--valid_freq', type=int, default=-1, help='frequency of validating the latest model')
        parser.add_argument('--valid_freq_ratio', type=float, default=1, help='calculating the valid_freq according to the ratio if valid_freq <= 0')
        parser.add_argument('--single_valid_freq_epoch', type=int, default=1, help='if exist, then valid every epoch after the number of epochs')
        parser.add_argument('--continue_train', action='store_true', default=False, help='continue training: load the latest model')
        parser.add_argument('--continue_epoch', type = str, default='optimal', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--op_name', type=str, default='SGD', help='# the name of optimizer')
        parser.add_argument('--niter', type=int, default=30, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--nesterov', type=bool, default=True, help='# nesterov')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
        parser.add_argument('--grad_clip_value', type=float, default=1, help='grad clip value')
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--momentum', type=float, default=0.9)

        #scheduler
        parser.add_argument('--lr_policy', type=str, default='mstep', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--with_warm_up', type=int, default=1)
        parser.add_argument('--warm_epoch', type=int, default=3)
        parser.add_argument('--lr_wait_epoch', type=int, default=3)
        parser.add_argument('--warm_start_factor', type=float, default=0.3)
        parser.add_argument('--lr_decay_factor', type=float, default=0.1)
        parser.add_argument('--lr_decay_iters', type=int, default=50)
        parser.add_argument('--patient_epoch', type=int, default=8)

        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = BaseOptions.gather_options(self)

        # get the basic options

        opt, _ = parser.parse_known_args()

        # modify sheduler-related parser options
        policy_name = opt.lr_policy
        if policy_name not in schedulers.basic_schedulers:
            policy_option_setter = schedulers.get_option_setter(policy_name)
            parser = policy_option_setter(parser)
        return parser

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        parser = BaseOptions.gather_options(self)
        return parser

