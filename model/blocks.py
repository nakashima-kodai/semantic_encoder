import torch
import torch.nn as nn
from . import utils
from . import normalizations


class ResBlocks(nn.Module):
    def __init__(self, nc, n_blocks, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()

        model = []
        for i in range(n_blocks):
            model += [ResBlock(nc, nc, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, dilation=1, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(input_nc, output_nc, 3, 1, dilation, norm, activation, pad_type, dilation=dilation)]
        model += [Conv2dBlock(output_nc, output_nc, 3, 1, dilation, norm, 'none', pad_type, dilation=dilation)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return x + self.model(x)

class ResBlock2(nn.Module):
    def __init__(self, input_nc, output_nc, dilation=1, norm='instance', activation='relu', pad_type='zero'):
        super(ResBlock2, self).__init__()

        model = []
        model += [Conv2dBlock(input_nc, output_nc, 3, 1, dilation, norm, activation, pad_type, dilation=dilation)]
        model += [Conv2dBlock(output_nc, output_nc, 3, 1, dilation, norm, 'none', pad_type, dilation=dilation)]
        self.model = nn.Sequential(*model)
        self.downsample = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(output_nc))

    def forward(self, x):
        output = self.model(x)
        output += self.downsample(x)
        return output

class downResBlock(nn.Module):
    def __init__(self, input_nc, output_nc, dilation=1, norm='instance', activation='relu', pad_type='zero'):
        super(downResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(input_nc, output_nc, 3, 2, dilation, norm, activation, pad_type, dilation=dilation)]
        model += [Conv2dBlock(output_nc, output_nc, 3, 1, dilation, norm, 'none', pad_type, dilation=dilation)]
        self.model = nn.Sequential(*model)
        self.downsample = nn.Sequential(nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=2, bias=False),
                                        nn.BatchNorm2d(output_nc))

    def forward(self, x):
        output = self.model(x)
        output += self.downsample(x)
        return output

class Conv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', dilation=1, bias=False):

        super(Conv2dBlock, self).__init__()

        padding = dilation if dilation != 1 else padding
        self.pad = utils.get_pad_layer(padding, pad_type)
        self.norm = utils.get_norm_layer(output_nc, norm)
        self.activation = utils.get_activation_layer(activation)

        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, dilation=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class upConv2dBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride=1, padding=0,
                 norm='none', activation='relu', pad_type='zero', bias=False):
        super(upConv2dBlock, self).__init__()

        self.pad = utils.get_pad_layer(padding, pad_type)
        self.norm = utils.get_norm_layer(output_nc, norm)
        self.activation = utils.get_activation_layer(activation)

        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size, stride, bias=bias)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm='none', activation='relu', bias=True):
        super(LinearBlock, self).__init__()

        self.fc = nn.Linear(input_nc, output_nc, bias=bias)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(output_nc)
        elif norm == 'instance':
            self.norm = nn.InstanceNorm1d(output_nc)
        elif norm == 'ln':
            self.norm = normalizations.LayerNorm(output_nc)
        elif norm == 'none':
            self.norm = None
        else:
            raise NotImplementedError('norm layer [{}] is not found'.format(norm_type))

        self.activation = utils.get_activation_layer(activation)

    def forward(self, x):
        x = self.fc(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x
