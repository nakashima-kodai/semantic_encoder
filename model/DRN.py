import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel
from .blocks import *
from . import networks


class DRN(BaseModel):
    def name(self):
        return 'DRN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.loss_names = ['drn']

        # set model
        self.model_names = ['drn', 'clf']
        self.drn = networks.DRNC26(opt.input_nc, 'batch', 'relu', 'zero')
        self.clf = nn.Conv2d(self.drn.output_nc, opt.n_class, kernel_size=1, bias=True)
        self.softmax = nn.LogSoftmax()

        if self.opt.phase == 'train':
            params = list(self.drn.parameters()) + list(self.clf.parameters())
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)

            self.criterion = nn.NLLLoss2d(ignore_index=255)

    def set_variables(self, data):
        self.image = data['image'].cuda()
        self.label = data['label'].cuda()

    def forward(self):
        hidden = self.drn(self.image)
        hidden = self.clf(hidden)
        hidden = F.interpolate(hidden, scale_factor=8, mode='bilinear')
        self.prob = self.softmax(hidden)

    def optimize_parameters(self):
        self.optimizer.zero_grad()

        self.forward()
        self.loss_drn = self.criterion(self.prob, self.label.long().squeeze(dim=1))

        self.loss_drn.backward()
        self.optimizer.step()

    def sample(self):
        colors_pred = []
        pred = torch.max(self.prob, dim=1)[1]
        for b in range(self.opt.batch_size):
            color_pred = self.trainid2color(pred[b].cpu().detach()).float() / 255.0
            colors_pred.append(color_pred.unsqueeze(dim=0))
        colors_pred = torch.cat(colors_pred, dim=0)

        image = (self.image.cpu() + 1.0) / 2.0
        saved_image = torch.cat((image, colors_pred), dim=0)
        return saved_image
