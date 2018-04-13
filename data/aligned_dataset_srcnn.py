import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from collections import OrderedDict
from PIL import ImageFilter

class AlignedDatasetSRCNN(BaseDataset):
	def initialize(self, opt):
		self.opt = opt
		self.root = opt.dataroot
		self.dir_A = os.path.join(opt.dataroot, opt.phase)
		self.dir_B = os.path.join(opt.dataroot, opt.phase+'annot')

		self.A_paths = sorted(make_dataset(self.dir_A))
		self.B_paths = sorted(make_dataset(self.dir_B))


		self.transform = get_transform(opt)

	def __getitem__(self, index):
		A_path = self.A_paths[index]
		B_path = self.B_paths[index]
		A = Image.open(A_path).convert('RGB')
		B = Image.open(B_path).convert('RGB')
		#encode color labeling to numbers
		B = self.encode_labelmap_color(B)	
		# make the image width loadSize
		ow, oh = A.size
		dw = self.opt.loadSize
		A = A.resize((dw, int(oh*dw/ow)), Image.BICUBIC)
		B = B.resize((dw, int(oh*dw/ow)), Image.NEAREST)

		# random scaling and rotation        
		if self.opt.phase == 'train':
			# random scale
			rand_scale = np.random.random_sample(1)*0.6+0.5
			ow2, oh2 = A.size
			A = A.resize((int(ow2*rand_scale), int(oh2*rand_scale)), Image.BICUBIC)
			B = B.resize((int(ow2*rand_scale), int(oh2*rand_scale)), Image.NEAREST)

			#pad with zero if too small
			ow3, oh3 = A.size
			if ow3<self.opt.fineSize or oh3<self.opt.fineSize:
				ow4 = self.opt.loadSize if self.opt.loadSize > ow3 else ow3
				oh4 = self.opt.loadSize if self.opt.loadSize > oh3 else oh3

				A_new = Image.new('RGB', (ow4, oh4), (0,0,0))
				B_new = Image.new('L', (ow4, oh4), 0)
				A_new.paste(A, (0,0,ow3,oh3))
				B_new.paste(B, (0,0,ow3,oh3))
			else:
				A_new = A
				B_new = B

			# random rotate
			rand_degree = np.random.randint(-60,60)
			A = A_new.rotate(rand_degree, resample=Image.BICUBIC)
			B = B_new.rotate(rand_degree, resample=Image.NEAREST)
		else:
			A = self.padding(A)
			B = self.padding(B)


		w, h = A.size

		

		#random crop and flip
		if self.opt.phase == 'train':
			w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
			h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))


			A = A.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))
			B = B.crop((w_offset, h_offset, w_offset + self.opt.fineSize, h_offset + self.opt.fineSize))


			if (not self.opt.no_flip) and random.random() < 0.5:
				A = A.transpose(Image.FLIP_LEFT_RIGHT)
				B = B.transpose(Image.FLIP_LEFT_RIGHT)




		A0 = A.copy()
		B0 = B.copy()
		A1 = A.copy()
		B1 = B.copy()
		A2 = A.copy()
		B2 = B.copy()

		if self.opt.phase == 'train':
			w_scale0 = int(self.opt.fineSize*0.25)
			w_scale1 = int(self.opt.fineSize*0.5)
			w_scale2 = int(self.opt.fineSize)

			h_scale0 = int(self.opt.fineSize*0.25)
			h_scale1 = int(self.opt.fineSize*0.5)
			h_scale2 = int(self.opt.fineSize)
		elif self.opt.phase == 'test':
			w_scale0 = int(w*0.25)
			w_scale1 = int(w*0.5)
			w_scale2 = int(w)

			h_scale0 = int(h*0.25)
			h_scale1 = int(h*0.5)
			h_scale2 = int(h)			

		A0 = A0.resize((w_scale0, h_scale0), Image.BICUBIC)
		B0 = B0.resize((w_scale0, h_scale0), Image.NEAREST)
		A1 = A1.resize((w_scale1, h_scale1), Image.BICUBIC)
		B1 = B1.resize((w_scale1, h_scale1), Image.NEAREST)		
		A2 = A2.resize((w_scale2, h_scale2), Image.BICUBIC)
		B2 = B2.resize((w_scale2, h_scale2), Image.NEAREST)	

		A0 = self.transform(A0)
		B0 = self.transform(B0)		
		A1 = self.transform(A1)
		B1 = self.transform(B1)	
		A2 = self.transform(A2)
		B2 = self.transform(B2)	

		# print(A0.size())
		# print(A1.size())
		# print(A2.size())


		return {'A0': A0, 'B0': B0,'A1': A1, 'B1': B1, 'A2': A2, 'B2': B2,
				'A_paths': A_path, 'B_paths': B_path}

	def __len__(self):
		return len(self.A_paths)

	def name(self):
		return 'Aligned Dataset of Scale RCNN'



	def padding(self, img_open):
		width, height = img_open.size
		if img_open.mode == 'RGB':
			img = Image.new('RGB', (self.ceil8(width), self.ceil8(height)), (0,0,0))
		elif img_open.mode == 'L':
			img = Image.new('L', (self.ceil8(width), self.ceil8(height)), 0)

		img.paste(img_open, (0,0,width,height))
		return img
		

	def ceil8(self, n):
		return int(np.ceil(n/16.0)*16)

	
	def encode_labelmap_color(self, labelmap, plot=False):
		labelmap = np.array(labelmap)
		dxchab_colors = OrderedDict([
			("Background", np.array([0, 0, 0], dtype=np.uint8)),
			("uac", np.array([0, 255, 0], dtype=np.uint8)),
			("ng", np.array([255, 0, 0], dtype=np.uint8)),
			("et", np.array([0, 0, 255], dtype=np.uint8)),
			("letter", np.array([255, 255, 0], dtype=np.uint8)),
		])
		im_out = (np.ones(labelmap.shape[:2]) * 255).astype(np.uint8)

		for gray_val, (label, rgb) in enumerate(dxchab_colors.items()):
			mask = np.all(labelmap == rgb, axis=-1)
			if label == 'Background':
				gray_val = 0
			elif (label == 'uac') or (label == 'ng') or (label == 'et'):
				gray_val = 1
			elif label =='letter':
				gray_val = 2
			im_out[mask] = gray_val

		return Image.fromarray(im_out)



