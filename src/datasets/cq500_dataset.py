import copy
import os
import json
import cv2
import numpy as np
import SimpleITK as sitk
import torch
import tqdm
from .ctbase_dataset import CTBaseDataset


class CQ500Dataset(CTBaseDataset):
    """
    Dataset for CQ500.
    """

    def __init__(self, opt, data_type):
        super().__init__(opt, data_type)

    def prepare_dataset(self):        
        self.dataset_0_json = []
        self.dataset_1_json = []

        with open(os.path.join(self.dataset_dir)) as f:
            self.dataset_json = json.load(f)[str(self.opt.target_focus)]
    
            for sample in tqdm.tqdm(self.dataset_json):
                if self.opt.target_focus in sample['label']:
                    self.dataset_1_json.append(sample)
                else:
                    self.dataset_0_json.append(sample)

        print('label 0 num: ', len(self.dataset_0_json))
        print('label 1 num: ', len(self.dataset_1_json))

        if self.opt.target_focus == 0:
            self.opt.data_norm_type = 'detection'
            self.mean = [123.675, 116.28, 103.53]
            self.std = [58.395, 57.12, 57.375]
            self.opt.edge_crop_sizes = [(30, 50),(55, 15),(60, 15)]
        else:
            self.opt.data_norm_type = 'competition'
            self.mean = [0.456, 0.456, 0.456]
            self.std = [0.224, 0.224, 0.224]
            self.opt.edge_crop_sizes = [(40, 40),(60, 60),(80, 80)]

    def read_dcm(self, p, **kwargs):
        itk_img = sitk.ReadImage(p)
        gray = sitk.GetArrayFromImage(itk_img)[0]

        scale1 = 1.0
        scale2 = 1.0
        if not gray.shape[0] == 512 or not gray.shape[1] == 512:
            scale1 = 512 / gray.shape[0]
            scale2 = 512 / gray.shape[1]
            gray = cv2.resize(gray, (512, 512))
        return gray, scale1, scale2

    def _load_sample(self, sample, idx = 0):
        x_list = []
        orig_x_list = []
        bbox_list = []
        positive_indices = []
        positive_image_indices = []

        gray_data = [self.read_dcm(p) for p in sample['dcm_paths']]
        gray_data = list(filter(lambda x: x is not None, gray_data))

        for img_ind in range(len(sample['dcm_paths'])):
            gray_list = []
            bag_is_positive = False

            gray, scale1, scale2 = gray_data[img_ind]
            gray_list.append(gray)

            bboxes = []
            if len(bboxes):
                positive_image_indices.append(img_ind)
            bbox_list.append(bboxes)

            if bag_is_positive:
                positive_indices.append(1)
            else:
                positive_indices.append(0)

            img = self.multiwindow_ww_wl(gray, self.multiwindows, gpu=False)
            img = self.norm_data(img)
            x_list.append(img)

        images = torch.stack(x_list, dim=0)
        positive_indices = np.array(positive_indices)

        if 'report' not in sample.keys():
            sample['report'] = ''

        positive_indices = positive_indices.astype('int')
        data = [images, sample['dcm_paths'][0], positive_indices, sample['report']]

        data.extend([bbox_list, -1])
        return data

    @staticmethod
    def collate_fn(data):
        collate_data = []
        for batch in data:
            for size0, size1 in batch[1]:
                l = copy.deepcopy(batch[0])
                l[0] = l[0][:, :, :-size0, size1: -size1]
                collate_data.append(l)
        return collate_data

    def __getitem__(self, idx):
        if idx < len(self.dataset_1_json):
            data = self._load_sample(self.dataset_1_json[idx], idx = idx)
            label = 1
        else:
            nidx = idx - len(self.dataset_1_json)
            data = self._load_sample(self.dataset_0_json[nidx], idx = nidx)
            label = 0
        data.extend([idx, label])
        return data, self.opt.edge_crop_sizes

    def __len__(self):
        return len(self.dataset_0_json) + len(self.dataset_1_json)