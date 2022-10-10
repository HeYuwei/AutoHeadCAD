import time
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import torch

from .vis_dataset import VisDataset
from util.basic import read_multi_data

class ScanDataset(VisDataset):
    """
    Dataset for train and test models.
    """

    def __init__(self, opt, task):
        super().__init__(opt, task)

    def _read_dcm_pydicom(self, p):
        chance = 2
        flag = False
        while chance > 0:
            chance -= 1
            try:
                ds = pydicom.dcmread(p)
                flag = True
                break
            except:
                time.sleep(1)
        if not flag:
            return None
        if hasattr(ds, 'SliceLocation'):
            sliceLocation = ds.SliceLocation
        else:
            return None
        gray = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        scale1 = 1.0
        scale2 = 1.0
        if not gray.shape[0] == 512 or not gray.shape[1] == 512:
            scale1 = 512 / gray.shape[0]
            scale2 = 512 / gray.shape[1]
            gray = cv2.resize(gray, (512, 512))
        return sliceLocation, gray, scale1, scale2

    def _load_series(self, sample):
        x_list = []
        vis_x_list = []
        dcm_paths = []

        data_list = read_multi_data(self._read_dcm_pydicom, sample['dcm_paths'], workers=5)
        loc_list = []
        for tmp_data in data_list:
            loc_list.append(tmp_data[0])
        arg_inds = np.argsort(loc_list).tolist()

        '''The parameter arg_inds guarantee'''
        gray_data = []
        for ind in arg_inds:
            if data_list[ind][1] is not None:
                gray_data.append(data_list[ind][1])
        num_images = len(gray_data)
        if num_images == 0:
            return None

        for image_idx in range(num_images):
            gray = gray_data[image_idx]
            if gray is None:
                continue
            dcm_paths.append(sample['dcm_paths'][arg_inds[image_idx]])

            img = self.multiwindow_ww_wl(gray, self.multiwindows)
            if self.opt.visualize and self.opt.vis_method is not None:
                vis_img = self.single_ww_wl(gray, self.opt.vis_ww, self.opt.vis_wl, gpu=False, jitter=False)
                vis_x_list.append(vis_img)

            img = self.norm_data(img)
            x_list.append(img)

        images = torch.stack(x_list, dim=0)
        data = [images, dcm_paths, -1]
        return data

    @staticmethod
    def collate_fn(data):
        return data

    def __getitem__(self, idx):
        if idx < len(self.positive_samples):
            sample, sample_idx = self.positive_samples[idx]
            data = self._load_series(sample)
            series_label = 1
        else:
            idx = idx - len(self.positive_samples)
            sample, sample_idx = self.negative_samples[idx]
            data = self._load_series(sample)
            series_label = 0
        
        data.append(series_label)
        data.append(sample_idx)
        return data

    def __len__(self):
        return len(self.positive_samples) + len(self.negative_samples)