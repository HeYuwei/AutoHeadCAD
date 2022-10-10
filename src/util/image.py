import numpy as np
from PIL import Image
import torch
import cv2

def norm2image(image,minp=None,maxp=None):
    if isinstance(minp, type(None)):
        minp = np.min(image)

    if isinstance(maxp, type(None)):
        maxp = np.max(image)

    image = image - minp
    maxp = maxp - minp

    image = image * (255 / maxp)
    image = image.astype('uint8')

    return image

def tensor2im(input_image, mode = 'gray', minp = None, maxp = None, normalize = True):
    """only process one image"""


    """"Converts a Tensor array into a numpy image array with C, H, W

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if isinstance(input_image, np.ndarray):
        image = input_image

    elif isinstance(input_image, torch.Tensor):  # get the data from a variable
        image = input_image.cpu().float().numpy()  # convert it into a numpy array

    image_shape = image.shape

    if len(image_shape) == 4:
        if image_shape[-1] == 1 or image_shape[-1] == 3:
            image = np.transpose(image,(0,3,1,2))

        if mode == 'gray':
            if image.shape[1] == 3 and image.shape[0] == 1:
                image = image[0]
                image = image.unsqueeze(1)
            else:
                assert False, 'the shape is error '

    elif len(image_shape) == 3:
        if image_shape[-1] == 1 or image_shape[-1] == 3:
            image = np.transpose(image,(2,0,1))

        elif mode == 'gray':
            image = np.expand_dims(image,1)


    elif len(image_shape) == 2:
        image = np.expand_dims(image, 0)

    if normalize:
        if isinstance(minp,type(None)):
            minp = np.min(image)

        if isinstance(maxp, type(None)):
            maxp = np.max(image)

        image = image - minp
        maxp = maxp - minp
    
        image = image * (255 / maxp)
        image = image.astype('uint8')
    return image

def cat_image(images,h_num = 10,factor = 0.75,in_type='HWC', out_type = 'HWC'):
    """

    Args:
        images: iamges with the same shape N x C x H x W or N x H x W x C
        h_num: images per line, -1 means all images in one line
        factor: the ratio factor for image size
        in_type: HWC or CHW
        out_type: HWC or CHW

    Returns:

    """

    if isinstance(images, list):
        images = np.stack(images, axis= 0)

    if in_type == 'HWC':
        images = np.transpose(images, (0, 3, 1, 2))

    if h_num == -1:
        h_num = images.shape[0]

    line_num = images.shape[0] // h_num
    rest_num = images.shape[0] - line_num * h_num

    image_data = []

    width = images.shape[-1]
    height = images.shape[-2]
    for i in range(line_num):
        image_line = np.zeros((images.shape[1],height,width*h_num))
        for j in range(h_num):
            image_line[:,:,j*width:(j+1)*width] = images[h_num*i + j]
        image_data.append(image_line)

    if rest_num > 0:
        last_line = np.zeros((images.shape[1],height,width*rest_num))

        for j in range(rest_num):
            last_line[:,:,j*width:(j+1)*width] = images[h_num*line_num + j]

        if line_num > 0:
            white = np.zeros((images.shape[1],height,width*(h_num-rest_num)))
            last_line = np.concatenate([last_line,white],axis=-1)

        image_data.append(last_line)
    n_image = np.concatenate(image_data,axis=1)
    n_image = n_image.transpose((1,2,0))
    # cv2.imwrite('test1.png', n_image[:,:,0])

    isize = (int(n_image.shape[1]*factor), int(n_image.shape[0]*factor))
    n_image = cv2.resize(n_image,isize)
    if len(n_image.shape) == 2:
        n_image = np.expand_dims(n_image,2)
    # cv2.imwrite('test2.png',n_image)
    if out_type == 'CHW':
        n_image = n_image.transpose((2,0,1))
    n_image = n_image.astype('uint8')
    return n_image

