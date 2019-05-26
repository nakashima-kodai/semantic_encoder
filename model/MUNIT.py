import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from . import utils


class MUNIT(BaseModel):
    def name(self):
        return 'MUNIT'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['gen', 'dis']

        # set generators
        self.model_names = ['gen_a', 'gen_b']
        self.gen_a = networks.AdaINGen(opt.input_nc, opt.ngf, opt.style_dim, opt.mlp_dim, n_down_s=4, n_down_c=2,
                                       n_res=4, activ='relu', pad_type='reflect')
        self.gen_b = networks.AdaINGen(opt.input_nc, opt.ngf, opt.style_dim, opt.mlp_dim, n_down_s=4, n_down_c=2,
                                       n_res=4, activ='relu', pad_type='reflect')

        if opt.isTrain:
            # set discriminators
            self.model_names += ['dis_a', 'dis_b']
            self.dis_a = networks.Discriminator(opt.crop_size, opt.input_nc, opt.ndf, opt.n_scale, n_layer=4,
                                                norm='none', activation='lrelu', pad_type='reflect', gan_type='lsgan')
            self.dis_b = networks.Discriminator(opt.crop_size, opt.input_nc, opt.ndf, opt.n_scale, n_layer=4,
                                                norm='none', activation='lrelu', pad_type='reflect', gan_type='lsgan')

            # set optimizers
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.paramsters())
            # self.optimizer_G = torch.optim.Adam(gen_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            # self.optimizer_D = torch.optim.Adam(dis_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_G = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_variables(self, src_data, tgt_data):
        self.image_src = src_data['image'].cuda()
        self.image_tgt = tgt_data['image'].cuda()

    def update_G(self):
        self.set_requires_grad([self.dis_a, self.dis_b], False)
        self.optimizer_G.zero_grad()



    def optimize_parameters(self):
        self.update_G()
