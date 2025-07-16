import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.scale_rcnn import SRCNN
import ntpath
from util import util
from PIL import Image
import io
from data.preprocess import preprocess_single_image


def segment_xray(image_bytes):
    """
    Function to segment X-ray images using a pre-trained SRCNN model.
    Args:
        image_bytes (bytes): The input image in bytes format.
    Returns:
        PIL.Image: The segmented image.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    model = SRCNN()
    model.initialize(opt)

    tensors = preprocess_single_image(image, fine_size=480)

    model.set_input(tensors)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    for label, image_numpy in visuals.items():
        print(image_numpy.shape)
        util.save_image(image_numpy, f"./test_output_{label}.png")


segment_xray(open("/Users/xin/Documents/past_research/test8.jpg", "rb").read())