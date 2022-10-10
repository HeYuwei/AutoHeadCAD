"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
from abc import ABC, abstractmethod
from imgaug import augmenters as iaa
from torchvision.transforms.functional import normalize
import torch

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt, data_type):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    @staticmethod
    def norm_data(x, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225), ctype = 'CHW', norm_to_one = True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()

        if ctype == 'HWC':
            x = x.permute(2,0,1)

        if norm_to_one: 
            x = x / 255.

        if x.shape[0] != 3:
            assert False, 'The dim is not right'
            
        x = normalize(x, mean, std)
        return x

    @staticmethod
    def get_transform(opt, dtype = 'CHW'):
        transform_list = []

        if 'resize' in opt.preprocess:
            t = iaa.Resize({"height": opt.load_size, "width": opt.load_size})
            transform_list.append(t)

        if 'c_crop' in opt.preprocess:
            t = iaa.CropToFixedSize(width=opt.crop_size, height=opt.crop_size, position='center')
            transform_list.append(t)

        elif 'u_crop' in opt.preprocess:
            t = iaa.CropToFixedSize(width=opt.crop_size, height=opt.crop_size, position='uniform')
            transform_list.append(t)

        if 'scale' in opt.preprocess:
            t = iaa.Affine(scale={"x": opt.scale_per_x, "y": opt.scale_per_y})
            t = iaa.Sometimes(opt.scale_rate,t)
            transform_list.append(t)

        if 'translate' in opt.preprocess:
            t = iaa.Affine(translate_px={"x": opt.translate_pix_x, "y": opt.translate_pix_y})
            t = iaa.Sometimes(opt.translate_rate, t)
            transform_list.append(t)

        if 'rotate' in opt.preprocess:
            t = iaa.Affine(rotate=opt.rotate_der)
            t = iaa.Sometimes(opt.rotate_rate, t)
            transform_list.append(t)

        if 'shear' in opt.preprocess:
            t = iaa.Affine(rotate=opt.shear_der)
            t = iaa.Sometimes(opt.shear_rate, t)
            transform_list.append(t)

        if 'elastic' in opt.preprocess:
            t = iaa.ElasticTransformation(alpha=opt.elastic_alpha, sigma=0.25)
            t = iaa.Sometimes(opt.elastic_rate, t)
            transform_list.append(t)

        if 'flip' in opt.preprocess:
            t = iaa.Fliplr(opt.flip_rate)
            transform_list.append(t)

        if 'contrast' in opt.preprocess:
            t = iaa.SigmoidContrast(gain=opt.contrast_gain, cutoff=opt.contrast_cutoff)
            t = iaa.Sometimes(opt.contrast_rate, t)
            transform_list.append(t)

        if 'clane' in opt.preprocess:
            t = iaa.CLAHE(clip_limit=opt.clane_limit)
            t = iaa.Sometimes(opt.clane_rate, t)
            transform_list.append(t)

        seq = iaa.Sequential(transform_list)


        def im_t(x):
            if dtype == 'CHW':
                return np.transpose(x,(1,2,0))
            return x

        trans = lambda x:np.transpose(seq.augment_image(image = im_t(x)),(2,0,1))

        return trans

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}



def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
