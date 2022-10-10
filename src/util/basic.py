"""This module contains simple helper functions """
from __future__ import print_function
import torch.multiprocessing as multiprocessing
import torch
import numpy as np
from PIL import Image
import os


def read_multi_data(func, param_list, workers = 3, ignore_none = True):
    import tqdm
    param_data = [[] for _ in range(workers)]
    c_worker = 0
    for i, param in enumerate(param_list):
        param_data[c_worker].append((i,param))
        c_worker = (c_worker + 1) % workers

    q = multiprocessing.Queue()
    q.cancel_join_thread()

    count = 0

    def read_data(func, param_part_list):

        if len(param_part_list) > 100:
            param_part_list = tqdm.tqdm(param_part_list)
        for i, param in param_part_list:
            try:
                if isinstance(param,list) and func.__code__.co_argcount > 1:
                    print(param)
                    data = func(*param)
                else:
                    data = func(param)
            except:
                data = None

            q.put((i, data))

    for i in range(workers):
        w = multiprocessing.Process(
            target=read_data,
            args=(func, param_data[i]))
        w.daemon = False
        w.start()

    data_list = [None for _ in range(len(param_list))]

    while count < len(param_list):
        i, data = q.get()
        data_list[i] = data
        count += 1

    new_data_list = []
    if ignore_none:
        for data in data_list:
            if data is not None:
                new_data_list.append(data)
        data_list = new_data_list

    return data_list

def numpy2table(cmatrix):
    table = '| |'
    for i in range(cmatrix.shape[0]):
        table += str(i) + '|'
    table += '\n|:-:|'
    for j in range(cmatrix.shape[0]):
        table += ':-:|'
    table += '\n'

    for i in range(cmatrix.shape[0]):
        table += '|' + str(i) + '|'
        for j in range(cmatrix.shape[0]):
            table += str(cmatrix[i,j]) + '|'
        table += '\n'

    return table

def dismantle_tuples(tuple_inputs):
    if not isinstance(tuple_inputs, list):
        return tuple_inputs
    new_list = []
    for t in tuple_inputs:
        new_list.append(t[0])
    return new_list

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """

    dirs = path.split('/')
    tmp_path = ''
    for d in dirs:
        tmp_path += d

        if len(d) > 0 and not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        tmp_path += '/'
