import os
import argparse
import torch
from utils.utils import mkdir


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment
        self.parser.add_argument('--model', type=str, default='MUNIT_semantic')
        self.parser.add_argument('--name', type=str, default='debug')
        self.parser.add_argument('--gpu_ids', type=str, default='0')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt')

        # dataset
        self.parser.add_argument('--source_dataset', type=str, default='gta5')
        self.parser.add_argument('--target_dataset', type=str, default='bdd100k')
        self.parser.add_argument('--source_dataroot', type=str, default='../../datasets/gta5')
        self.parser.add_argument('--target_dataroot', type=str, default='../../datasets/bdd100k')
        self.parser.add_argument('--n_class', type=int, default=19)

        # data_loader
        self.parser.add_argument('--load_size', type=int, default=512)
        self.parser.add_argument('--crop_size', type=int, default=256)
        self.parser.add_argument('--input_nc', type=int, default=3)
        self.parser.add_argument('--scale_transform', type=str, default='scale_width_and_crop', choices=['resize', 'resize_and_crop', 'crop', 'scale_width', 'scale_width_and_crop'])
        self.parser.add_argument('--no_shuffle', action='store_true')
        self.parser.add_argument('--no_flip', action='store_true')

        # generator
        self.parser.add_argument('--ngf', type=int, default=64)
        self.parser.add_argument('--style_dim', type=int, default=8)
        self.parser.add_argument('--mlp_dim', type=int, default=256)

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        opt = self.parser.parse_args()
        opt.isTrain = self.isTrain

        # print options to the terminal
        args = vars(opt)
        print('--------------- Options ---------------')
        for k, v in sorted(args.items()):
            print('{:<21} : {}'.format(str(k), str(v)))
        print('----------------- End -----------------\n')

        # make directory
        if self.isTrain:
            ckpt_dir = os.path.join(opt.ckpt_dir, opt.name)
            sample_dir = os.path.join(opt.sample_dir, opt.name)
            mkdir(ckpt_dir)
            mkdir(sample_dir)

            # save options to the disk
            file_path = os.path.join(ckpt_dir, 'options.txt')
            with open(file_path, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('{:<21} : {}\n'.format(str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        else:
            result_dir = os.path.join(opt.result_dir, opt.name)
            mkdir(result_dir)

        # set CUDA device
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
            # torch.backends.cudnn.benchmark=True

        return opt
