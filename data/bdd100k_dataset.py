import os
import torch
from torch.utils import data
from PIL import Image
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class BDD100KDataset(data.Dataset):
    def __init__(self, opt, source=True):
        self.opt = opt
        dataroot = opt.source_dataroot if source else opt.target_dataroot
        phase = 'train' if opt.isTrain else 'test'

        dir_img = os.path.join(dataroot, phase+'_img')
        dir_lbl = os.path.join(dataroot, phase+'_label')

        self.paths_img = make_dataset(dir_img)
        self.paths_lbl = make_dataset(dir_lbl)

    def name(self):
        return 'BDD100KDataset'

    def __getitem__(self, index):
        # load image
        path_img = self.paths_img[index]
        image = Image.open(path_img).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_img = get_transform(self.opt, params)
        image = transform_img(image)

        # load label
        path_lbl = self.paths_lbl[index]
        label = Image.open(path_lbl)
        transform_lbl = get_transform(self.opt, params, Image.NEAREST, normalize=False)
        label = transform_lbl(label)

        input_dict = {'image': image, 'label': label}
        return input_dict

    def __len__(self):
        return len(self.paths_img)
