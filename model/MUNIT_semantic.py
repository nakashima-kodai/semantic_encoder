import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from . import utils


class MUNIT_semantic(BaseModel):
    def name(self):
        return 'MUNIT_semantic'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.style_dim = opt.style_dim

        # self.loss_names = ['gen', 'dis', 'clf', 'gen_rec_a', 'gen_rec_b', 'gen_rec_s_a', 'gen_rec_s_b',
        #                    'gen_rec_c_a', 'gen_rec_c_b', 'gen_cyc_a', 'gen_cyc_b', 'gen_adv_a', 'gen_adv_b',
        #                    'gen_sem_a', 'gen_sem_b', 'dis_a', 'dis_b']

        self.loss_names = ['gen', 'dis', 'clf', 'gen_rec_a', 'gen_rec_b',
                           'gen_cyc_a', 'gen_cyc_b', 'gen_adv_a', 'gen_adv_b',
                           'gen_sem_a', 'gen_sem_b', 'dis_a', 'dis_b']

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

            # set classifier
            self.model_names += ['drn', 'clf', 'softmax']
            self.drn = networks.DRNC26(opt.input_nc, 'batch', 'relu', 'zero')
            self.clf = nn.Conv2d(self.drn.output_nc, opt.n_class, kernel_size=1, bias=True)
            self.softmax = nn.LogSoftmax()

            # set optimizers
            gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters())
            dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
            clf_params = list(self.drn.parameters()) + list(self.clf.parameters())
            # self.optimizer_G = torch.optim.Adam(gen_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            # self.optimizer_D = torch.optim.Adam(dis_params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_G = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_D = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizer_C = torch.optim.SGD(clf_params, lr=0.01, momentum=0.9, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_C)

            # set criterion
            self.criterion_clf = nn.NLLLoss2d(ignore_index=255)

            # random noise
            self.s_a_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()
            self.s_b_rnd = torch.randn(opt.batch_size, opt.style_dim, 1, 1).cuda()

    def criterion_rec(self, input, target):
        return torch.mean(torch.abs(input - target))

    def classification(self, image):
        hidden = self.drn(image)
        hidden = self.clf(hidden)
        hidden = F.interpolate(hidden, scale_factor=8, mode='bilinear')
        return self.softmax(hidden)

    def set_variables(self, src_data, tgt_data):
        self.image_a = src_data['image'].cuda()
        self.image_b = tgt_data['image'].cuda()
        self.label_a = src_data['label'].cuda()
        self.label_b = tgt_data['label']  # Not used in training
        self.color_a = src_data['color']  # Not used in training
        self.color_b = tgt_data['color']  # Not used in training

    def optimize_parameters(self):
        self.update_C()
        self.update_D()
        self.update_G()

    def update_C(self):
        self.set_requires_grad([self.drn, self.clf], True)
        self.optimizer_C.zero_grad()

        pred_image_a = self.classification(self.image_a)
        self.loss_clf = self.criterion_clf(pred_image_a, self.label_a.long().squeeze(dim=1))

        self.loss_clf.backward()
        self.optimizer_C.step()

    def update_D(self):
        self.set_requires_grad([self.dis_a, self.dis_b], True)
        self.optimizer_D.zero_grad()

        s_a_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()
        s_b_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()

        # encode
        _, c_a = self.gen_a.encode(self.image_a)
        _, c_b = self.gen_b.encode(self.image_b)

        # decode (translation)
        image_a_fake = self.gen_a.decode(s_a_rnd, c_b)
        image_b_fake = self.gen_b.decode(s_b_rnd, c_a)

        # compute discriminator losses
        self.loss_dis_a = self.dis_a.calc_dis_loss(image_a_fake.detach(), self.image_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(image_b_fake.detach(), self.image_b)
        self.loss_dis = self.loss_dis_a + self.loss_dis_b

        self.loss_dis.backward()
        self.optimizer_D.step()

    def update_G(self):
        self.set_requires_grad([self.dis_a, self.dis_b, self.drn, self.clf], False)
        self.optimizer_G.zero_grad()

        s_a_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()
        s_b_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()

        # encode
        s_a, c_a = self.gen_a.encode(self.image_a)
        s_b, c_b = self.gen_b.encode(self.image_b)

        # decode (reconstruction)
        image_a_rec = self.gen_a.decode(s_a, c_a)
        image_b_rec = self.gen_b.decode(s_b, c_b)

        # decode (translation)
        image_a_fake = self.gen_a.decode(s_a_rnd, c_b)
        image_b_fake = self.gen_b.decode(s_b_rnd, c_a)

        # encode again
        s_a_rnd_rec, c_b_rec = self.gen_a.encode(image_a_fake)
        s_b_rnd_rec, c_a_rec = self.gen_b.encode(image_b_fake)

        # decode again
        image_a_cyc = self.gen_a.decode(s_a, c_a_rec)
        image_b_cyc = self.gen_b.decode(s_b, c_b_rec)

        # prediction of fake image
        pseudo_label_b = torch.max(self.classification(self.image_b), dim=1)[1]
        pred_fake_a = self.classification(image_a_fake)
        pred_fake_b = self.classification(image_b_fake)

        # compute generator losses
        self.loss_gen_rec_a = self.criterion_rec(image_a_rec, self.image_a)
        self.loss_gen_rec_b = self.criterion_rec(image_b_rec, self.image_b)
        # self.loss_gen_rec_s_a = self.criterion_rec(s_a_rnd_rec, s_a_rnd)
        # self.loss_gen_rec_s_b = self.criterion_rec(s_b_rnd_rec, s_b_rnd)
        # self.loss_gen_rec_c_a = self.criterion_rec(c_a_rec, c_a)
        # self.loss_gen_rec_c_b = self.criterion_rec(c_b_rec, c_b)
        self.loss_gen_cyc_a = self.criterion_rec(image_a_cyc, self.image_a)
        self.loss_gen_cyc_b = self.criterion_rec(image_b_cyc, self.image_b)
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(image_a_fake)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(image_b_fake)
        self.loss_gen_sem_a = self.criterion_clf(pred_fake_a, self.label_a.long().squeeze(dim=1))
        self.loss_gen_sem_b = self.criterion_clf(pred_fake_b, pseudo_label_b)
        # self.loss_gen = self.opt.lambda_rec * (self.loss_gen_rec_a + self.loss_gen_rec_b) + \
        #                 self.loss_gen_rec_s_a + self.loss_gen_rec_s_b + \
        #                 self.loss_gen_rec_c_a + self.loss_gen_rec_c_b + \
        #                 self.opt.lambda_rec * (self.loss_gen_cyc_a + self.loss_gen_cyc_b) + \
        #                 self.loss_gen_adv_a + self.loss_gen_adv_b + \
        #                 self.loss_gen_sem_a + self.loss_gen_sem_b

        self.loss_gen = self.opt.lambda_rec * (self.loss_gen_rec_a + self.loss_gen_rec_b) + \
                        self.opt.lambda_rec * (self.loss_gen_cyc_a + self.loss_gen_cyc_b) + \
                        self.loss_gen_adv_a + self.loss_gen_adv_b + \
                        self.loss_gen_sem_a + self.loss_gen_sem_b

        self.loss_gen.backward()
        self.optimizer_G.step()

    def sample(self):
        self.eval_mode()

        s_a_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()
        s_b_rnd = torch.randn(self.opt.batch_size, self.style_dim, 1, 1).cuda()

        saved_image = [self.image_a.cpu(), self.image_b.cpu()]
        with torch.no_grad():
            # encode
            s_a, c_a = self.gen_a.encode(self.image_a)
            s_b, c_b = self.gen_b.encode(self.image_b)

            # decode (reconstruction)
            image_a_rec = self.gen_a.decode(s_a, c_a)
            image_b_rec = self.gen_b.decode(s_b, c_b)
            saved_image += [image_a_rec.cpu(), image_b_rec.cpu()]

            # decode (translation)
            image_a_fake1 = self.gen_a.decode(self.s_a_rnd, c_b)
            image_b_fake1 = self.gen_b.decode(self.s_b_rnd, c_a)
            image_a_fake2 = self.gen_a.decode(s_a_rnd, c_b)
            image_b_fake2 = self.gen_b.decode(s_b_rnd, c_a)
            saved_image += [image_b_fake1.cpu(), image_a_fake1.cpu(), image_b_fake2.cpu(), image_a_fake2.cpu()]

            # prediction of fake image
            pseudo_label_a = torch.max(self.classification(self.image_a), dim=1)[1]
            pseudo_label_b = torch.max(self.classification(self.image_b), dim=1)[1]
            label_fake_a = torch.max(self.classification(image_a_fake1), dim=1)[1]
            label_fake_b = torch.max(self.classification(image_b_fake1), dim=1)[1]

            saved_image += [self.color_a, self.color_b]
            saved_image += self.trainid2color_batch(pseudo_label_a.cpu().detach(), self.opt.batch_size)
            saved_image += self.trainid2color_batch(pseudo_label_b.cpu().detach(), self.opt.batch_size)
            saved_image += self.trainid2color_batch(label_fake_a.cpu().detach(), self.opt.batch_size)
            saved_image += self.trainid2color_batch(label_fake_b.cpu().detach(), self.opt.batch_size)

        saved_image = torch.cat(saved_image, dim=0)
        return saved_image
