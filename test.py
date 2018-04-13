import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from util.visualizer import Visualizer
from util import html
from models.scale_rcnn import SRCNN


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = SRCNN()
model.initialize(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    if opt.sourceoftest == 'internal':
        visualizer.save_images(webpage, visuals, img_path)
    elif opt.sourceoftest == 'external':
        visualizer.save_images_nogt(webpage, visuals, img_path)

webpage.save()
