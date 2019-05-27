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

        self.loss_names = ['gen', 'dis', 'gen_rec_src', 'gen_rec_tgt', 'gen_rec_s_src', 'gen_rec_s_tgt',
                           'gen_rec_c_src', 'gen_rec_c_tgt', 'gen_adv_src', 'gen_adv_tgt', 'dis_src', 'dis_tgt']

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
            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            self.optimizer_G = torch.optim.Adam(gen_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(dis_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # random noise
            self.s_src_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()
            self.s_tgt_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()


    def criterion_rec(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_variables(self, src_data, tgt_data):
        self.image_src = src_data['image'].cuda()
        self.image_tgt = tgt_data['image'].cuda()

    def update_G(self):
        self.set_requires_grad([self.dis_a, self.dis_b], False)
        self.optimizer_G.zero_grad()

        s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
        s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

        # encode
        s_src_real, c_src_real = self.gen_a.encode(self.image_src)
        s_tgt_real, c_tgt_real = self.gen_b.encode(self.image_tgt)

        # decode (reconstruction)
        rec_image_src = self.gen_a.decode(s_src_real, c_src_real)
        rec_image_tgt = self.gen_b.decode(s_tgt_real, c_tgt_real)

        # decode (translation)
        fake_src = self.gen_a.decode(s_src_rnd, c_tgt_real)
        fake_tgt = self.gen_b.decode(s_tgt_rnd, c_src_real)

        # encode again
        rec_s_src_rnd, rec_c_tgt_real = self.gen_a.encode(fake_src)
        rec_s_tgt_rnd, rec_c_src_real = self.gen_b.encode(fake_tgt)

        # calculate loss
        self.loss_gen_rec_src = self.criterion_rec(rec_image_src, self.image_src)
        self.loss_gen_rec_tgt = self.criterion_rec(rec_image_tgt, self.image_tgt)
        self.loss_gen_rec_s_src = self.criterion_rec(rec_s_src_rnd, s_src_rnd)
        self.loss_gen_rec_s_tgt = self.criterion_rec(rec_s_tgt_rnd, s_tgt_rnd)
        self.loss_gen_rec_c_src = self.criterion_rec(rec_c_src_real, c_src_real)
        self.loss_gen_rec_c_tgt = self.criterion_rec(rec_c_tgt_real, c_tgt_real)
        self.loss_gen_adv_src = self.dis_a.calc_gen_loss(fake_src)
        self.loss_gen_adv_tgt = self.dis_b.calc_gen_loss(fake_tgt)
        self.loss_gen = self.opt.lambda_rec_image * (self.loss_gen_rec_src + self.loss_gen_rec_tgt) + \
                        self.opt.lambda_rec_s * (self.loss_gen_rec_s_src + self.loss_gen_rec_c_src) + \
                        self.opt.lambda_rec_c * (self.loss_gen_rec_c_src + self.loss_gen_rec_c_tgt) + \
                        self.opt.lambda_adv * (self.loss_gen_adv_src + self.loss_gen_adv_tgt)

        self.loss_gen.backward()
        self.optimizer_G.step()

    def update_D(self):
        self.set_requires_grad([self.dis_a, self.dis_b], True)
        self.optimizer_D.zero_grad()

        s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
        s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

        # encode
        s_src_real, c_src_real = self.gen_a.encode(self.image_src)
        s_tgt_real, c_tgt_real = self.gen_b.encode(self.image_tgt)

        # decode (translation)
        fake_src = self.gen_a.decode(s_src_rnd, c_tgt_real)
        fake_tgt = self.gen_b.decode(s_tgt_rnd, c_src_real)

        self.loss_dis_src = self.dis_a.calc_dis_loss(fake_src.detach(), self.image_src)
        self.loss_dis_tgt = self.dis_b.calc_dis_loss(fake_tgt.detach(), self.image_tgt)
        self.loss_dis = self.loss_dis_src + self.loss_dis_tgt

        self.loss_dis.backward()
        self.optimizer_D.step()

    def optimize_parameters(self):
        self.update_D()
        self.update_G()
        torch.cuda.synchronize()

    def forward(self):
        self.eval_mode()

        with torch.no_grad():
            s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
            s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

            # encode
            s_src_real, c_src_real = self.gen_a.encode(self.image_src)
            s_tgt_real, c_tgt_real = self.gen_b.encode(self.image_tgt)

            # decode (reconstruction)
            rec_image_src = self.gen_a.decode(s_src_real, c_src_real)
            rec_image_tgt = self.gen_b.decode(s_tgt_real, c_tgt_real)

            # decode (translation)
            fake_src1 = self.gen_a.decode(s_src_rnd, c_tgt_real)
            fake_tgt1 = self.gen_b.decode(s_tgt_rnd, c_src_real)

            fake_src2 = self.gen_a.decode(self.s_src_rnd, c_tgt_real)
            fake_tgt2 = self.gen_b.decode(self.s_tgt_rnd, c_src_real)

        self.train_mode()

        # de-normalization
        img_src = (self.image_src.cpu() + 1.0) / 2.0
        img_tgt = (self.image_tgt.cpu() + 1.0) / 2.0
        rec_src = (rec_image_src.cpu() + 1.0) / 2.0
        rec_tgt = (rec_image_tgt.cpu() + 1.0) / 2.0
        fake_src1 = (fake_src1.cpu() + 1.0) / 2.0
        fake_src2 = (fake_src2.cpu() + 1.0) / 2.0
        fake_tgt1 = (fake_tgt1.cpu() + 1.0) / 2.0
        fake_tgt2 = (fake_tgt2.cpu() + 1.0) / 2.0

        saved_image = torch.cat((img_src, img_tgt, rec_src, rec_tgt, fake_tgt1, fake_src1, fake_tgt2, fake_src2), dim=0)
        return saved_image
