import os
import torch
from torch.utils import data
from PIL import Image
from .base_dataset import BaseDataset, get_params, get_transform
from .image_folder import make_dataset


class BDD100KDataset(data.Dataset):
    def __init__(self, opt, is_source=True):
        self.opt = opt
        dataroot = opt.source_dataroot if is_source else opt.target_dataroot
        phase = 'train' if opt.isTrain else 'test'

        dir_image = os.path.join(dataroot, phase+'_img')
        dir_label = os.path.join(dataroot, phase+'_label')
        dir_color = os.path.join(dataroot, phase+'_color')

        self.paths_image = sorted(make_dataset(dir_image))
        self.paths_label = sorted(make_dataset(dir_label))
        self.paths_color = sorted(make_dataset(dir_color))

    def name(self):
        return 'BDD100KDataset'

    def __getitem__(self, index):
        # load image
        path_image = self.paths_image[index]
        image = Image.open(path_image).convert('RGB')
        params = get_params(self.opt, image.size)
        transform_image = get_transform(self.opt, params)
        image = transform_image(image)

        # load label
        path_label = self.paths_label[index]
        label = Image.open(path_label).convert('L')
        transform_label = get_transform(self.opt, params, Image.NEAREST, normalize=False)
        label = transform_label(label) * 255

        # load color label
        path_color = self.paths_color[index]
        color = Image.open(path_color).convert('RGB')
        color = transform_label(color)

        input_dict = {'image': image, 'label': label, 'color': color}
        return input_dict

    def __len__(self):
        return len(self.paths_image)
