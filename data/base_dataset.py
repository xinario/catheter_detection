import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.NEAREST))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale':
        transform_list.append(transforms.Lambda(
            lambda img: __scale(img, opt.loadSize)))
    elif opt.resize_or_crop == 'scaleboth':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.NEAREST))
    elif opt.resize_or_crop == 'original':
        transform_list.append(transforms.Lambda(
            lambda img: __padding(img)))
    elif opt.resize_or_crop == 'none':
        pass


    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.NEAREST)


def __scale(img, target_size):
    ow, oh = img.size
    omin = min(ow, oh)
    if (omin > target_size):
        return img
    elif (omin == ow):
        w = target_size
        h = int(target_size * oh / ow)
        return img.resize((w, h), Image.NEAREST)
    elif (omin == oh):
        h = target_size
        w = int(target_size * ow / oh)
        return img.resize((w, h), Image.NEAREST)


def __padding(img_open):
    width, height = img_open.size
    if img_open.mode == 'RGB':
        img = Image.new('RGB', (ceil8(width), ceil8(height)), (0,0,0))
    elif img_open.mode == 'L':
        img = Image.new('L', (ceil8(width), ceil8(height)), 0)

    img.paste(img_open, (0,0,width,height))
    return img
    

def ceil8(n):
    return int(np.ceil(n/16.0)*16)