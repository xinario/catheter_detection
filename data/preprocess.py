# preprocess.py
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import (Compose, Resize, RandomCrop, ToTensor,
                                    Normalize, InterpolationMode, Lambda)
def ceil8(n):
    return int(np.ceil(n/16.0)*16)

def padding(img_open):
    width, height = img_open.size
    if img_open.mode == 'RGB':
        img = Image.new('RGB', (ceil8(width), ceil8(height)), (0,0,0))
    elif img_open.mode == 'L':
        img = Image.new('L', (ceil8(width), ceil8(height)), 0)

    img.paste(img_open, (0,0,width,height))
    return img

def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

def preprocess_single_image(image: Image.Image, fine_size=256):
    # Convert and resize image to three scales
    image = image.convert("RGB")
    dw = 480
    ow, oh = image.size
    image = image.resize((dw, int(oh*dw/ow)), Image.BICUBIC)
    image = padding(image)
    w, h = image.size

    w0, h0 = int(w * 0.25), int(h * 0.25)
    w1, h1 = int(w * 0.5), int(h * 0.5)
    w2, h2 = w, h

    scales = [(w0, h0), (w1, h1), (w2, h2)]
    A0 = get_transform()(image.resize(scales[0], Image.BICUBIC))
    A1 = get_transform()(image.resize(scales[1], Image.BICUBIC))
    A2 = get_transform()(image.resize(scales[2], Image.BICUBIC))

    return {
        "A0": A0.unsqueeze(0),
        "A1": A1.unsqueeze(0),
        "A2": A2.unsqueeze(0),
        "B0": A0.unsqueeze(0),  # Assuming B is the same as A for testing
        "B1": A1.unsqueeze(0),
        "B2": A2.unsqueeze(0),
        "A_paths": ["dummy_path"],  # Placeholder for image paths
        "B_paths": ["dummy_path"],  # Placeholder for image paths
    }