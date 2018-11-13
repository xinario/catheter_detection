import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
class SRCNN(BaseModel):
    def name(self):
        return 'Scale Recurrent CNN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A0 = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.fineSize/4), int(opt.fineSize/4))
        self.input_B0 = self.Tensor(opt.batchSize, opt.output_nc,
                                   int(opt.fineSize/4), int(opt.fineSize/4))
        self.input_A1 = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.fineSize/2), int(opt.fineSize/2))
        self.input_B1 = self.Tensor(opt.batchSize, opt.output_nc,
                                   int(opt.fineSize/2), int(opt.fineSize/2))

        self.input_A2 = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B2 = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.nclass = opt.output_nc
        # load/define networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.output_nc)

               
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)



        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions

            self.criterion = util.cross_entropy2d

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)

        print('-----------------------------------------------')

    def set_input(self, input):
        input_A0 = input['A0']
        input_B0 = input['B0']
        input_A1 = input['A1']
        input_B1 = input['B1']
        input_A2 = input['A2']
        input_B2 = input['B2']

        self.input_A0.resize_(input_A0.size()).copy_(input_A0)
        self.input_B0.resize_(input_B0.size()).copy_(input_B0)
        self.input_A1.resize_(input_A1.size()).copy_(input_A1)
        self.input_B1.resize_(input_B1.size()).copy_(input_B1)
        self.input_A2.resize_(input_A2.size()).copy_(input_A2)
        self.input_B2.resize_(input_B2.size()).copy_(input_B2)


        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A0 = self.input_A0
        self.real_A1 = self.input_A1
        self.real_A2 = self.input_A2

        input_list = [self.real_A0, self.real_A1, self.real_A2]

        output_list = self.netG(input_list)

        self.fake_B0 = output_list[0]
        self.fake_B1 = output_list[1]
        self.fake_B2 = output_list[2]


        self.real_B0 = self.input_B0
        self.real_B1 = self.input_B1
        self.real_B2 = self.input_B2
    # no backprop gradients
    def test(self):
        with torch.no_grad():
            self.real_A0 = self.input_A0
            self.real_A1 = self.input_A1
            self.real_A2 = self.input_A2

            input_list = [self.real_A0, self.real_A1, self.real_A2]

            output_list = self.netG(input_list)

            self.fake_B0 = output_list[0]
            self.fake_B1 = output_list[1]
            self.fake_B2 = output_list[2]


            self.real_B0 = self.input_B0
            self.real_B1 = self.input_B1
            self.real_B2 = self.input_B2
    # get image paths
    def get_image_paths(self):
        return self.image_paths



    def backward_G(self):


        self.loss_G_scale0 = self.criterion(self.fake_B0[:,0:5,:,:], self.real_B0, weight=torch.Tensor([1, 40, 80]).cuda())
        self.loss_G_scale1 = self.criterion(self.fake_B1[:,0:5,:,:], self.real_B1, weight=torch.Tensor([1, 40, 80]).cuda())
        self.loss_G_scale2 = self.criterion(self.fake_B2[:,0:5,:,:], self.real_B2, weight=torch.Tensor([1, 40, 80]).cuda())


        self.loss_G = self.loss_G_scale0 + self.loss_G_scale1 + self.loss_G_scale2 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()


        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_scale0', self.loss_G_scale0.data[0]),
                            ('G_scale1', self.loss_G_scale1.data[0]),
                            ('G_scale2', self.loss_G_scale2.data[0]),
                            ])

    def get_current_visuals(self):
        real_A0 = util.tensor2im(self.real_A0.data)
        real_A1 = util.tensor2im(self.real_A1.data)
        real_A2 = util.tensor2im(self.real_A2.data)

        # original
        fake_B0 = util.tensor2im_segmap(F.softmax(self.fake_B0, dim=1).data[:,[0,1,2],:,:])
        fake_B1 = util.tensor2im_segmap(F.softmax(self.fake_B1, dim=1).data[:,[0,1,2],:,:])
        fake_B2 = util.tensor2im_segmap(F.softmax(self.fake_B2, dim=1).data[:,[0,1,2],:,:])

        real_B0 = util.tensor2im(self.one2multimap(self.real_B0).data[:,[0,1,2],:,:])
        real_B1 = util.tensor2im(self.one2multimap(self.real_B1).data[:,[0,1,2],:,:])
        real_B2 = util.tensor2im(self.one2multimap(self.real_B2).data[:,[0,1,2],:,:])

        return OrderedDict([('real_A0', real_A0), ('real_A1', real_A1), ('real_A2', real_A2), ('fake_B0', fake_B0), ('fake_B1', fake_B1), ('fake_B2', fake_B2), ('real_B0', real_B0), ('real_B1', real_B1), ('real_B2', real_B2)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

    def multimap2one(self, map):
        map = map.data.cpu().numpy()
        #0, 1,...c
        map = np.argmax(map, axis=1).astype(np.float32)
        #normalize to -1, 1
        map = (map/255)*2 - 1
        map = Variable(torch.Tensor(map).unsqueeze(1).cuda())
        return map


    def one2multimap(self, map):
        map = map.squeeze(dim=1)
        map = map.data.cpu().numpy()
        map = np.round((map+1)*0.5*255)
        map = map.astype(np.int32)
        output_map = []
        for i in range(self.nclass):
            tmp = np.zeros(map.shape)
            tmp[map==i] = 1
            output_map.append(tmp)

        out = np.stack(output_map, 1).astype(np.float32)
        out = 2*out-1

        out = Variable(torch.Tensor(out).cuda())
        return out
