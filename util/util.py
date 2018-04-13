from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import torch.nn.functional as F
from collections import OrderedDict

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
	# CHW
	image_numpy = image_tensor[0].cpu().float().numpy()
	image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
	return image_numpy.astype(imtype)


def tensor2im_segmap(image_tensor, imtype=np.uint8):
	# CHW
	#for fcn
	image_numpy = image_tensor[0].cpu().float().numpy()
	if image_numpy.shape[0] == 1:
		image_numpy = decode_segmap_color(image_numpy)
		image_numpy = np.transpose(image_numpy, (1, 2, 0))
	elif image_numpy.shape[0] == 2:
		image_numpy = np.tile(image_numpy[1], (3, 1, 1))
		image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
	elif image_numpy.shape[0] == 3:
		image_numpy = np.transpose(image_numpy, (1, 2, 0))  * 255.0
	return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
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
	image_pil = Image.fromarray(image_numpy)
	image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
	"""Print methods and doc strings.
	Takes module, class, list, dictionary, or string."""
	methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
	processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
	print( "\n".join(["%s %s" %
					 (method.ljust(spacing),
					  processFunc(str(getattr(object, method).__doc__)))
					 for method in methodList]) )

def varname(p):
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m:
			return m.group(1)

def print_numpy(x, val=True, shp=False):
	x = x.astype(np.float64)
	if shp:
		print('shape,', x.shape)
	if val:
		x = x.flatten()
		print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
			np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
	if isinstance(paths, list) and not isinstance(paths, str):
		for path in paths:
			mkdir(path)
	else:
		mkdir(paths)


def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)



def cross_entropy2d(input, target, weight=None, size_average=True):
	n, c, h, w = input.size()
	target = (target+1)*0.5*255
	# print(torch.max(target))
	target = torch.round(target).type(torch.cuda.LongTensor)


	log_p = F.log_softmax(input, dim=1)
	log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
	log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
	log_p = log_p.view(-1, c)

	mask = target >= 0
	target = target[mask]

	loss = F.nll_loss(log_p, target, ignore_index=250,
					  weight=weight, size_average=False)
	if size_average:
		loss /= mask.data.sum()
	return loss




def decode_segmap_color(segmap):
	# -1,1
	segmap = segmap.squeeze().astype(np.float32)
	# 0-255
	segmap = np.round((segmap+1)*0.5*255.0).astype(np.uint8)

	dxchab_colors = OrderedDict([
		("Background", np.array([0, 0, 0], dtype=np.uint8)),
		("line", np.array([0, 255, 0], dtype=np.uint8)),
		("letter", np.array([255, 255, 0], dtype=np.uint8)),
	])

	im_out = (np.ones((3, segmap.shape[0],segmap.shape[1] )) * 0).astype(np.uint8)
	r, g, b = im_out[0,:,:], im_out[1,:,:], im_out[2,:,:]

	for gray_val, (label, rgb) in enumerate(dxchab_colors.items()):
		match_pxls = (segmap == gray_val)


		r[match_pxls] = rgb[0]
		g[match_pxls] = rgb[1]
		b[match_pxls] = rgb[2]

	return im_out