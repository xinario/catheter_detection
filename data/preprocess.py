# preprocess.py
import numpy as np
from PIL import Image
from torchvision import transforms

def get_preprocessing_transforms(fine_size):
    return transforms.Compose([
        transforms.Resize((fine_size, fine_size), Image.BICUBIC),
        transforms.ToTensor()
    ])