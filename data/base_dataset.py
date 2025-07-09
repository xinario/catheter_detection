import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

from torchvision.transforms import (Compose, Resize, RandomCrop, ToTensor,
                                    Normalize, InterpolationMode, Lambda)
from functools import partial

import numpy as np

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def scale_width(img, target):
    ow, oh = img.size
    if ow == target:
        return img
    return img.resize((target, int(target*oh/ow)),
                      InterpolationMode.BICUBIC)
def to_long(t):
    return t.long()
# -----------------------------------------------


def get_transform(opt, *, for_label=False):
    tfms = []

    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        interp = (InterpolationMode.NEAREST
                  if for_label else InterpolationMode.BICUBIC)
        tfms += [Resize(osize, interpolation=interp),
                 RandomCrop(opt.fineSize)]

    elif opt.resize_or_crop == 'scale_width':
        tfms.append(Lambda(partial(scale_width, target=opt.fineSize)))

    tfms.append(ToTensor())

    if for_label:
        tfms.append(Lambda(to_long))            # <- no nested lambda
    else:
        tfms.append(Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5)))

    return Compose(tfms)