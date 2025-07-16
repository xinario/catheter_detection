import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.scale_rcnn import SRCNN
import ntpath
from util import util
from PIL import Image
import io
from data.preprocess import get_preprocessing_transforms


fine_size = 512  # or whatever your fineSize was during training
transform = get_preprocessing_transforms(fine_size)

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

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = SRCNN()
    model.initialize(opt)

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('%04d: process image... %s' % (i, img_path))
        image_dir = "./"
        short_path = ntpath.basename(img_path[0])
        name = os.path.splitext(short_path)[0]
        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

