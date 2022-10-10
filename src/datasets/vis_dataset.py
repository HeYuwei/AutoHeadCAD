import json
import os
import time
import cv2
import numpy as np
import pydicom
import tqdm
import torch
import torch.multiprocessing as multiprocessing
from .ctbase_dataset import CTBaseDataset


class VisDataset(CTBaseDataset):
    """
    Dataset for visualization.
    """
    def __init__(self, opt, task):
        super().__init__(opt, task)
        
    def prepare_dataset(self):         
        self.dataset_0_json = []
        self.dataset_1_json = []

        if self.dataset_dir.endswith('.json'):
            with open(os.path.join(self.dataset_dir)) as f:
                self.dataset_json = json.load(f)
            for sample_idx, sample in tqdm.tqdm(enumerate(self.dataset_json)):
                if isinstance(sample['label'], list):
                    flag = self.opt.target_focus in sample['label']
                elif isinstance(sample['label'], int):
                    flag = self.opt.target_focus == sample['label']
                if flag:
                    self.dataset_1_json.append((sample, sample_idx))
                else:
                    self.dataset_0_json.append((sample, sample_idx))
        else:
            series_dir_list = list(filter(lambda x: not x.endswith('.json'), os.listdir(self.dataset_dir)))
            for series_dir in tqdm.tqdm(series_dir_list):
                split = series_dir.split('_')
                sample_idx = int(split[0])
                label = int(split[-1])
                num_images = len(os.listdir(os.path.join(self.dataset_dir, series_dir))) - 1
                sample = {'series_dir': os.path.join(self.dataset_dir, series_dir)}
                
                if self.opt.target_focus == label:
                    self.dataset_1_json.append((sample, sample_idx))
                else:
                    self.dataset_0_json.append((sample, sample_idx))

        self.prepare_new_epoch()

    def prepare_new_epoch(self):
        self.positive_samples = self.dataset_1_json
        self.negative_samples = self.dataset_0_json

    def _read_files(self, dcm_paths, workers=3):
        path_data = [[] for _ in range(workers)]
        c_worker = 0
        for i, path in enumerate(dcm_paths):
            path_data[c_worker].append((i,path))
            c_worker = (c_worker + 1) % workers

        def read_dcm(q, dcm_parts):
            for i, p in dcm_parts:
                while True:
                    try:
                        ds = pydicom.dcmread(p)
                        break
                    except:
                        print(p)
                        time.sleep(1)
                        continue

                gray = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
                scale1 = 1.0
                scale2 = 1.0
                if not gray.shape[0] == 512 or not gray.shape[1] == 512:
                    scale1 = 512 / gray.shape[0]
                    scale2 = 512 / gray.shape[1]
                    gray = cv2.resize(gray, (512, 512))
                q.put((i, gray, scale1, scale2))

        queue = multiprocessing.Queue()
        queue.cancel_join_thread()

        for i in range(workers):
            w = multiprocessing.Process(
                target=read_dcm,
                args=(queue, path_data[i]))
            w.daemon = False
            w.start()

        data_dict = {}
        while len(data_dict.keys()) < len(dcm_paths):
            i, gray, scale1, scale2 = queue.get()
            data_dict[str(i)] = (gray, scale1, scale2)
        return data_dict

    def _load_series(self, sample):
        x_list = []
        labels = []
        vis_x_list = []
        bbox_list = []
        positive_bag_indices = []
        positive_image_indices = []
        bag_dcm_paths = []

        record_file = os.path.join(sample['series_dir'], 'record.json')
        with open(record_file, 'r') as f:
            obj = json.load(f)

        num_images = len(obj['bbox_list'])
        bbox_list = obj['bbox_list'] if self.opt.target_focus in obj['label'] else [[] for i in range(num_images)]
        dcm_paths = [os.path.join(sample['series_dir'], '{}.dcm'.format(idx)) for idx in range(num_images)]

        gray_data = self._read_files(dcm_paths, workers=15)
        for image_idx in range(num_images):
            gray_list = []
            gray, scale1, scale2 = gray_data[str(image_idx)]
            # annotation
            if self.opt.target_focus in obj['label'] and len(bbox_list[image_idx]) > 0:
                positive_image_indices.append(image_idx)
                positive_bag_indices.append(image_idx)
                labels.append(1)
            else:
                labels.append(0)
                
            bag_dcm_paths.append(dcm_paths[image_idx])
            # ww and wl
            img = self.multiwindow_ww_wl(gray, self.multiwindows)
            if self.opt.visualize and self.opt.vis_method is not None:
                vis_img = self.single_ww_wl(gray, self.opt.vis_ww, self.opt.vis_wl, gpu=False, jitter=False)
                vis_x_list.append(vis_img)
            
            # normalize
            img = self.norm_data(img)
            x_list.append(img)

        images = torch.stack(x_list, dim=0)
        data = [images, positive_bag_indices, positive_image_indices, bag_dcm_paths, -1]

        if self.opt.visualize and self.opt.vis_method is not None:
            orig_images = torch.stack(vis_x_list, dim=0)
            data.extend([bbox_list, orig_images])

        data.append(np.array(labels))
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
        return len(self.positive_samples)
