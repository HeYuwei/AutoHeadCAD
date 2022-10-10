from abc import abstractmethod
import random
import time

from imgaug import augmenters as iaa
import numpy as np
import pydicom

from .base_dataset import BaseDataset
from util import *


class CTBaseDataset(BaseDataset):
    """
    Base dataset class for CT task.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def __init__(self, opt, task):
        super().__init__(opt, task)

        print('target focus:', opt.target_focus)
        assert task in ['train','valid', 'test']
        self.task = task
        self.target_focus = opt.target_focus
        self.adjust = iaa.Resize({"height": 512, "width": 512})
        self.multiwindows = opt.multiwindows[self.target_focus]
        if self.opt.data_norm_type == 'detection':
            self.mean = [123.675, 116.28, 103.53]
            self.std = [58.395, 57.12, 57.375]
        elif self.opt.data_norm_type == 'competition':
            self.mean = [0.456, 0.456, 0.456]
            self.std = [0.224, 0.224, 0.224]
        if self.task in ['test']:
            self.name = 'valid'
        else:
            self.name = self.task
        self.dataset_dir = getattr(self.opt, self.name+'_dataset_dir')
        self.prepare_dataset()

    @abstractmethod
    def prepare_dataset(self):
        pass

    def prepare_new_epoch(self):
        pass

    def _read_gray(self, dcm_path):
        while True:
            try:
                ds = pydicom.read_file(dcm_path)
                break
            except:
                time.sleep(10)
        gray = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        if not (gray.shape[0] == 512 and gray.shape[1] == 512):
            gray = self.adjust.augment_image(image=gray)
        return gray

    def norm_data(self, x):
        x = x.float()
        if self.opt.data_norm_type in ['gray',' normal', 'activitynet', 'kinetics', 'competition']:
            x = x / 255.
        if self.mean and self.std:
            x = self._normalize(x, self.mean, self.std, inplace=True)
        return x

    def _normalize(self, tensor, mean, std, inplace=True):
        """Normalize a tensor image with mean and standard deviation.

        .. note::
            This transform acts out of place by default, i.e., it does not mutates the input tensor.

        See :class:`~torchvision.transforms.Normalize` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace(bool,optional): Bool to make this operation inplace.

        Returns:
            Tensor: Normalized Tensor image.
        """

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
        if tensor.shape[0] == 3:
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif tensor.shape[0] == 1:
            tensor.sub_(mean[0, None, None]).div_(std[0, None, None])
        else:
            raise AssertionError('invalid number of image channels')
        return tensor

    def single_ww_wl(self, gray, ww, wl, gpu=True, jitter=False):
        img = torch.from_numpy(gray).float().clone()
        if gpu:
            img = img.cuda(self.opt.gpu_ids[0])
        if jitter:
            c_ww = random.sample(range(int(ww - 0.1 * ww), int(ww + 0.1 * ww)),1)[0]
            c_wl = random.sample(range(int(wl - 0.1 * wl), int(wl + 0.1 * wl)),1)[0]
            ww, wl = c_ww, c_wl
        minv = wl - ww
        maxv = wl + ww
        dn = maxv - minv
        img.clamp_(minv, maxv)
        img.sub_(minv).mul_(255.0 / dn)
        img = img.repeat(3,1,1)
        return img

    def multiwindow_ww_wl(self, gray, ww_wl_list, gpu=True, jitter=False):
        img = torch.from_numpy(gray).float().clone()
        if gpu:
            img = img.cuda(self.opt.gpu_ids[0])
        img = img.repeat(3,1,1)
        for i in range(3):
            ww = ww_wl_list[i][0]
            wl = ww_wl_list[i][1]
            if jitter:
                c_ww = random.sample(range(int(ww - 0.1 * ww), int(ww + 0.1 * ww)),1)[0]
                c_wl = random.sample(range(int(wl - 0.1 * wl), int(wl + 0.1 * wl)),1)[0]
                ww, wl = c_ww, c_wl
            minv = wl - ww
            maxv = wl + ww
            dn = maxv - minv
            img[i] = img[i].clamp_(minv, maxv)
            img[i] = img[i].sub_(minv).mul_(255.0 / dn)
        return img

    @staticmethod
    def collate_fn(data):
        pass

    def get_collate_fn(self):
        return self.collate_fn

    @abstractmethod
    def __len__(self):
        pass
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

    
