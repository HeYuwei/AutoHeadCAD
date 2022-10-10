import numpy as np
import argparse
import pydicom
import argparse
import json
import os

def mk_dataset(label_path, data_dir = '../data/cq500', save_dir='../data'):
    '''

    Args:
        label_path: the labels information downloaed with DCM data together
        data_dir: the data dir saving the decompressed cq500
        save_dir: the dir to save processed labels information, which is saved in json format

    Returns:

    We extracted the shortest normal CT scan from each study.

    '''

    blood_inds = []
    cbone_inds = []

    with open(label_path) as f:
        lines = f.readlines()

        keys = lines[0].split(',')
        for i, k in enumerate(keys):
            if ':ICH' in k:
                blood_inds.append(i)
            if ':CalvarialFracture' in k:
                cbone_inds.append(i)

    inds_data = [blood_inds, cbone_inds]
    label_data = [[],[]]

    for line in lines[1:]:
        items = line.split(',')
        seq_id = items[0].split('-')[-1]

        for atype, inds in enumerate(inds_data):
            label_count = [0,0]
            for ind in inds:
                tmp_label = int(items[ind])
                label_count[tmp_label] += 1

            if label_count[1] > label_count[0]:
                label_data[atype].append(seq_id)

    blood_label_info = set(label_data[0])
    cbone_label_info = set(label_data[1])


    def read_item(sid):
        d_path = os.path.join(data_dir, 'CQ500CT' + sid + ' ' + 'CQ500CT' + sid)
        if not os.path.exists(d_path):
            return None

        tmp_dict = {}
        label_list = []
        tmp_path_data = []
        tmp_path_lens = []

        for r, ds, fs in os.walk(d_path):
            fs = list(filter(lambda x: x.lower().endswith('.dcm'), fs))

            '''Remove scans with abnormal length'''
            if len(fs) >= 15 and len(fs) <= 80:
                tmp_paths = [os.path.join(r, f) for f in fs]
                tmp_path_data.append(tmp_paths)
                tmp_path_lens.append(len(tmp_paths))

        if len(tmp_path_lens) == 0:
            return None

        min_ind = np.argmin(tmp_path_lens).item()

        dcm_paths = tmp_path_data[min_ind]
        ds_list = [pydicom.dcmread(tmp_path) for tmp_path in dcm_paths]
        loc_list = [ds.SliceLocation for ds in ds_list]
        arg_inds = np.argsort(loc_list).tolist()
        dcm_paths = [dcm_paths[ind] for ind in arg_inds]

        if sid in blood_label_info:
            label_list.append(0)

        if sid in cbone_label_info:
            label_list.append(2)

        tmp_dict['dcm_paths'] = dcm_paths
        tmp_dict['label'] = label_list
        tmp_dict['bbox_paths'] = None

        return tmp_dict

    data_list = []
    tmp_count = 0
    for sid in range(491):
        item = read_item(str(sid))
        if item is not None:
            data_list.append(item)
        else:
            print(sid)
            tmp_count += 1

    print('abnormal count ', tmp_count)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with open(os.path.join(save_dir, 'cq500.json'),'w') as f:
        json.dump(data_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', default='../data/cq500', type= str, help= 'the folder path saving downloaded *.zip and read.csv')
    opt = parser.parse_args()
    mk_dataset(label_path=os.path.join(opt.p, 'reads.csv'))
