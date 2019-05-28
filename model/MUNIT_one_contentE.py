import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from . import utils


class MUNIT_one_contentE(BaseModel):
    def name(self):
        return 'MUNIT_one_contentE'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['gen', 'dis', 'gen_rec_src', 'gen_rec_tgt', 'gen_rec_s_src', 'gen_rec_s_tgt',
                           'gen_rec_c_src', 'gen_rec_c_tgt', 'gen_adv_src', 'gen_adv_tgt', 'dis_src', 'dis_tgt']

        # set generators
        self.model_names = ['style_enc_src', 'style_enc_tgt', 'content_enc',
                            'dec_src', 'dec_tgt', 'mlp_src', 'mlp_tgt']
        n_down_s = 4
        n_down_c = 2
        n_res = 4
        self.style_enc_src = networks.StyleEncoder(opt.input_nc, opt.ngf, opt.style_dim, n_down_s, 'none', 'relu', 'reflect')
        self.style_enc_tgt = networks.StyleEncoder(opt.input_nc, opt.ngf, opt.style_dim, n_down_s, 'none', 'relu', 'reflect')
        self.content_enc = networks.ContentEncoder(opt.input_nc, opt.ngf, n_down_c, n_res, 'instance', 'relu', 'reflect')
        self.dec_src = networks.Decoder(self.content_enc.output_nc, opt.input_nc, n_down_c, n_res, 'ln', 'relu', 'reflect')
        self.dec_tgt = networks.Decoder(self.content_enc.output_nc, opt.input_nc, n_down_c, n_res, 'ln', 'relu', 'reflect')

        n_adain_params = self.get_num_adain_params(self.dec_src)
        print('num_adain_params: {}'.format(n_adain_params))
        mlp_blocks = 3
        self.mlp_src = networks.MLP(opt.style_dim, n_adain_params, opt.mlp_dim, mlp_blocks, 'none', 'relu')
        self.mlp_tgt = networks.MLP(opt.style_dim, n_adain_params, opt.mlp_dim, mlp_blocks, 'none', 'relu')

        if opt.isTrain:
            # set discriminators
            self.model_names += ['dis_src', 'dis_tgt']
            self.dis_src = networks.Discriminator(opt.crop_size, opt.input_nc, opt.ndf, opt.n_scale, n_layer=4,
                                                  norm='none', activation='lrelu', pad_type='reflect', gan_type='lsgan')
            self.dis_tgt = networks.Discriminator(opt.crop_size, opt.input_nc, opt.ndf, opt.n_scale, n_layer=4,
                                                  norm='none', activation='lrelu', pad_type='reflect', gan_type='lsgan')

            # set optimizers
            gen_params = list(self.style_enc_src.parameters()) + list(self.style_enc_tgt.parameters()) + \
                         list(self.content_enc.parameters()) + \
                         list(self.dec_src.parameters()) + list(self.dec_tgt.parameters()) + \
                         list(self.mlp_src.parameters()) + list(self.mlp_tgt.parameters())
            dis_params = list(self.dis_src.parameters()) + list(self.dis_tgt.parameters())
            self.optimizer_G = torch.optim.Adam(gen_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam(dis_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # random noise
            self.s_src_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()
            self.s_tgt_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

    def criterion_rec(self, input, target):
        return torch.mean(torch.abs(input - target))

    def set_variables(self, src_data, tgt_data):
        self.image_src = src_data['image'].cuda()
        self.image_tgt = tgt_data['image'].cuda()

    def optimize_parameters(self):
        self.update_D()
        self.update_G()

    def update_D(self):
        self.set_requires_grad([self.dis_src, self.dis_tgt], True)
        self.optimizer_D.zero_grad()

        s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
        s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

        # encode
        c_src = self.content_enc(self.image_src)
        c_tgt = self.content_enc(self.image_tgt)

        # decode (translation)
        adain_params_src = self.mlp_src(s_src_rnd)
        self.assign_adain_params(adain_params_src, self.dec_src)
        fake_src = self.dec_src(c_tgt)

        adain_params_tgt = self.mlp_tgt(s_tgt_rnd)
        self.assign_adain_params(adain_params_tgt, self.dec_tgt)
        fake_tgt = self.dec_tgt(c_src)

        self.loss_dis_src = self.dis_src.calc_dis_loss(fake_src.detach(), self.image_src)
        self.loss_dis_tgt = self.dis_tgt.calc_dis_loss(fake_tgt.detach(), self.image_tgt)
        self.loss_dis = self.loss_dis_src + self.loss_dis_tgt

        self.loss_dis.backward()
        self.optimizer_D.step()

    def update_G(self):
        self.set_requires_grad([self.dis_src, self.dis_tgt], False)
        self.optimizer_G.zero_grad()

        s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
        s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

        # encode
        s_src = self.style_enc_src(self.image_src)
        s_tgt = self.style_enc_tgt(self.image_tgt)
        c_src = self.content_enc(self.image_src)
        c_tgt = self.content_enc(self.image_tgt)

        # decode (reconstruction)
        adain_params_src = self.mlp_src(s_src)
        self.assign_adain_params(adain_params_src, self.dec_src)
        rec_image_src = self.dec_src(c_src)

        adain_params_tgt = self.mlp_tgt(s_tgt)
        self.assign_adain_params(adain_params_tgt, self.dec_tgt)
        rec_image_tgt = self.dec_tgt(c_tgt)

        # decode (translation)
        adain_params_src = self.mlp_src(s_src_rnd)
        self.assign_adain_params(adain_params_src, self.dec_src)
        fake_src = self.dec_src(c_tgt)

        adain_params_tgt = self.mlp_tgt(s_tgt_rnd)
        self.assign_adain_params(adain_params_tgt, self.dec_tgt)
        fake_tgt = self.dec_tgt(c_src)

        # encode again
        rec_s_src_rnd = self.style_enc_src(fake_src)
        rec_s_tgt_rnd = self.style_enc_tgt(fake_tgt)
        rec_c_src = self.content_enc(fake_tgt)
        rec_c_tgt = self.content_enc(fake_src)

        # compute losses
        self.loss_gen_rec_src = self.criterion_rec(rec_image_src, self.image_src)
        self.loss_gen_rec_tgt = self.criterion_rec(rec_image_tgt, self.image_tgt)
        self.loss_gen_rec_s_src = self.criterion_rec(rec_s_src_rnd, s_src_rnd)
        self.loss_gen_rec_s_tgt = self.criterion_rec(rec_s_tgt_rnd, s_tgt_rnd)
        self.loss_gen_rec_c_src = self.criterion_rec(rec_c_src, c_src)
        self.loss_gen_rec_c_tgt = self.criterion_rec(rec_c_tgt, c_tgt)
        self.loss_gen_adv_src = self.dis_src.calc_gen_loss(fake_src)
        self.loss_gen_adv_tgt = self.dis_tgt.calc_gen_loss(fake_tgt)
        self.loss_gen = self.opt.lambda_rec * (self.loss_gen_rec_src + self.loss_gen_rec_tgt) + \
                        self.loss_gen_rec_s_src + self.loss_gen_rec_s_tgt + \
                        self.loss_gen_rec_c_src + self.loss_gen_rec_c_tgt + \
                        self.loss_gen_adv_src + self.loss_gen_adv_tgt

        self.loss_gen.backward()
        self.optimizer_G.step()

    def sample(self):
        self.eval_mode()

        with torch.no_grad():
            s_src_rnd = torch.randn(self.image_src.size(0), self.opt.style_dim, 1, 1).cuda()
            s_tgt_rnd = torch.randn(self.image_tgt.size(0), self.opt.style_dim, 1, 1).cuda()

            # encode
            s_src = self.style_enc_src(self.image_src)
            s_tgt = self.style_enc_tgt(self.image_tgt)
            c_src = self.content_enc(self.image_src)
            c_tgt = self.content_enc(self.image_tgt)

            # decode (reconstruction)
            adain_params_src = self.mlp_src(s_src)
            self.assign_adain_params(adain_params_src, self.dec_src)
            rec_image_src = self.dec_src(c_src)

            adain_params_tgt = self.mlp_tgt(s_tgt)
            self.assign_adain_params(adain_params_tgt, self.dec_tgt)
            rec_image_tgt = self.dec_tgt(c_tgt)

            # decode (translation)
            adain_params_src = self.mlp_src(s_src_rnd)
            self.assign_adain_params(adain_params_src, self.dec_src)
            fake_src1 = self.dec_src(c_tgt)

            adain_params_tgt = self.mlp_tgt(s_tgt_rnd)
            self.assign_adain_params(adain_params_tgt, self.dec_tgt)
            fake_tgt1 = self.dec_tgt(c_src)

            adain_params_src = self.mlp_src(self.s_src_rnd)
            self.assign_adain_params(adain_params_src, self.dec_src)
            fake_src2 = self.dec_src(c_tgt)

            adain_params_tgt = self.mlp_tgt(self.s_tgt_rnd)
            self.assign_adain_params(adain_params_tgt, self.dec_tgt)
            fake_tgt2 = self.dec_tgt(c_src)

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
