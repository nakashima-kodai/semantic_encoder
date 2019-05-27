import os
import numpy as np
import torch
from PIL import Image
from collections import OrderedDict
from . import utils


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.save_dir = os.path.join(opt.ckpt_dir, opt.name)

        self.model_names = []
        self.loss_names = []
        self.optimizers = []

        self.epoch_losses = OrderedDict()

    def setup(self):
        # set schedulers
        if self.opt.isTrain:
            self.schedulers = [utils.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]

        # load or initialize networks
        if self.opt.isTrain:
            self.init_networks()
        else:
            self.load_networks(self.opt.load_epoch)

        # save networks architecture to disk
        self.print_networks()

        # networks to cuda
        if len(self.opt.gpu_ids):
            self.to_cuda()

        # reset epoch loss dictionary
        self.reset_epoch_losses()

    def init_networks(self):
        for name in self.model_names:
            print('{:>3} : '.format(name), end='')
            net = getattr(self, name)
            utils.init_weights(net, self.opt.init_type)

    def to_cuda(self):
        for name in self.model_names:
            net = getattr(self, name)
            setattr(self, name, net.cuda())

    def update_lr(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def eval_mode(self):
        for name in self.model_names:
            net = getattr(self, name)
            net.eval()

    def train_mode(self):
        for name in self.model_names:
            net = getattr(self, name)
            net.train()

    def save_networks(self, epoch):
        for name in self.model_names:
            save_filename = '%s_%s.pth' % (epoch, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, name)

            print('saving the model to {}'.format(save_path))
            if len(self.opt.gpu_ids):
                torch.save(net.cpu().state_dict(), save_path)
                net.cuda()
            else:
                torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, epoch, save_dir=None):
        for name in self.model_names:
            load_filename = '%s_%s.pth' % (epoch, name)
            if save_dir is None:
                save_dir = self.save_dir
            load_path = os.path.join(save_dir, load_filename)
            net = getattr(self, name)

            print('loading the model from {}'.format(load_path))
            state_dict = torch.load(load_path)
            net.load_state_dict(state_dict)

    def print_networks(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
            save_path = os.path.join(self.save_dir, name + '.txt')
            with open(save_path, 'wt') as f:
                f.write(str(net))
                f.write('\nTotal number of parameters: {}'.format(num_params))

    def get_current_losses(self):
        loss_dict = OrderedDict()
        for name in self.loss_names:
            loss_dict[name] = float(getattr(self, 'loss_'+name).item())
        return loss_dict

    def reset_epoch_losses(self):
        for name in self.loss_names:
            self.epoch_losses[name] = 0.0

    def sum_epoch_losses(self):
        for name in self.loss_names:
            self.epoch_losses[name] += float(getattr(self, 'loss_'+name).item())

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def trainid2color(self, label):
        color = torch.from_numpy(CITYSCAPE_PALETTE[label].transpose((2, 0, 1)))
        return color
