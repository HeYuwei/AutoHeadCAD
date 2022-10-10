import json
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import warnings
warnings.filterwarnings("ignore")

from datasets import create_dataset
from models import create_model
from options.base_options import TestOptions
from util.visualizer import Visualizer


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.valid_dataset_dir = '../data/cq500.json'
    opt.load_dir = os.path.join(opt.checkpoints_dir, str(opt.target_focus))
    opt.vis_dir = os.path.join(opt.vis_dir, str(opt.target_focus))

    visualizer = Visualizer(opt, opt.l_state)
    v_dataset = create_dataset(opt, opt.l_state)

    model = create_model(opt)
    model.setup(opt)
    model.save_dir = os.path.join('../cq500_stat_info', str(opt.target_focus))
    if not os.path.exists(model.save_dir):
        os.makedirs(model.save_dir)
    model.validation(v_dataset, visualizer, valid_iter=-1)