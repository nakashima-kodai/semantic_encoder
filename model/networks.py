import torch
from torch import nn
from .blocks import *

################################################################################
# Discriminator
################################################################################
class Discriminator(nn.Module):
    def __init__(self, image_size, input_nc, ndf, n_scale, n_layer, norm, activation, pad_type, gan_type):
        super(Discriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layer = n_layer
        self.norm = norm
        self.activation = activation
        self.pad_type = pad_type
        self.gan_type = gan_type

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(n_scale):
            self.cnns.append(self.make_net())

    def make_net(self):
        ndf = self.ndf

        cnn_x = [Conv2dBlock(self.input_nc, ndf, 4, 2, 1, norm='none', activation=self.activation, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(ndf, 2*ndf, 4, 2, 1, norm=self.norm, activation=self.activation, pad_type=self.pad_type)]
            ndf *= 2
        cnn_x += [nn.Conv2d(ndf, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs_fake = self.forward(input_fake)
        outs_real = self.forward(input_real)
        loss = 0

        for it, (out_f, out_r) in enumerate(zip(outs_fake, outs_real)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_f - 0)**2) + torch.mean((out_r - 1)**2)  # original MUNIT implementation
                # loss += 0.5 * (torch.mean((out_f - 0)**2) + torch.mean((out_r - 1)**2))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs_fake = self.forward(input_fake)
        loss = 0
        for it, (out_f) in enumerate(outs_fake):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out_f - 1)**2)  # original MUNIT implementation
                # loss += 0.5 * torch.mean((out_f - 1)**2)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

################################################################################
# Generator
################################################################################
class AdaINGen(nn.Module):
    def __init__(self, input_nc, ngf, style_dim, mlp_dim, n_down_s, n_down_c, n_res, activ, pad_type):
        super(AdaINGen, self).__init__()

        self.enc_style = StyleEncoder(input_nc, ngf, style_dim, n_down_s, 'none', activ, pad_type)
        self.enc_content = ContentEncoder(input_nc, ngf, n_down_c, n_res, 'instance', activ, pad_type)
        self.dec = Decoder(self.enc_content.output_nc, input_nc, n_down_c, n_res, 'ln', activ, pad_type)

        n_adain_params = self.get_num_adain_params(self.dec)
        print('num_adain_params: {}'.format(n_adain_params))
        mlp_blocks = 3
        self.mlp = MLP(style_dim, n_adain_params, mlp_dim, mlp_blocks, 'none', activ)

    def forward(self, x):
        style, content = self.encode(x)
        image_rec = self.decode(style, content)
        return image_rec

    def encode(self, x):
        style = self.enc_style(x)
        content = self.enc_content(x)
        return style, content

    def decode(self, style, content):
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        image = self.dec(content)
        return image

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                # print('adain_params: {}, mean: {}, std: {}'.format(adain_params.size(), mean.size(), std.size()))
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

################################################################################
# Encoder and Decoder
################################################################################
class StyleEncoder(nn.Module):
    def __init__(self, input_nc, ngf, style_dim, n_down, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()

        model = [Conv2dBlock(input_nc, ngf, 7, 1, 3, norm, activ, pad_type)]
        for i in range(n_down):
            mult = 2**i
            i_c = min(mult*ngf, 256)
            o_c = min(2*i_c, 256)
            model += [Conv2dBlock(i_c, o_c, 4, 2, 1, norm, activ, pad_type)]

        model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        model += [nn.Conv2d(o_c, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class ContentEncoder(nn.Module):
    def __init__(self, input_nc, ngf, n_down, n_res, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()

        model = [Conv2dBlock(input_nc, ngf, 7, 1, 3, norm, activ, pad_type)]
        for i in range(n_down):
            mult = 2**i
            model += [Conv2dBlock(mult*ngf, 2*mult*ngf, 4, 2, 1, norm, activ, pad_type)]
        model += [ResBlocks(2*mult*ngf, n_res, norm, activ, pad_type)]
        self.model = nn.Sequential(*model)
        self.output_nc = 2*mult*ngf

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_up, n_res, norm, activ, pad_type):
        super(Decoder, self).__init__()

        model = [ResBlocks(input_nc, n_res, 'adain', activ, pad_type)]
        for i in range(n_up):
            model += [upConv2dBlock(input_nc, input_nc//2, 5, 1, 2, norm, activ, pad_type)]
            input_nc //= 2
        model += [Conv2dBlock(input_nc, output_nc, 7, 1, 3, 'none', 'tanh', pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

################################################################################
# MLP
################################################################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_blocks, norm, activ):
        super(MLP, self).__init__()

        model = [LinearBlock(input_dim, hidden_dim, norm, activ)]
        for i in range(n_blocks-2):
            model += [LinearBlock(hidden_dim, hidden_dim, norm, activ)]
        model += [LinearBlock(hidden_dim, output_dim, 'none', 'none')]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

################################################################################
# semantic segmentation model
################################################################################
class DRNC26(nn.Module):
    def __init__(self, input_nc, norm, activation, pad_type):
        super(DRNC26, self).__init__()

        # level 1
        model = [Conv2dBlock(input_nc, 16, 7, 1, 3, norm, activation, pad_type)]
        model += [ResBlock(input_nc=16, output_nc=16, norm=norm)]
        # level 2
        model += [downResBlock(input_nc=16, output_nc=32, norm=norm)]
        # level 3
        model += [downResBlock(input_nc=32, output_nc=64, norm=norm)]
        model += [ResBlock(input_nc=64, output_nc=64, norm=norm)]
        # level 4
        model += [downResBlock(input_nc=64, output_nc=128, norm=norm)]
        model += [ResBlock(input_nc=128, output_nc=128, norm=norm)]
        # level 5
        model += [ResBlock2(input_nc=128, output_nc=256, dilation=2, norm=norm)]
        model += [ResBlock(input_nc=256, output_nc=256, dilation=2, norm=norm)]
        # level 6
        model += [ResBlock2(input_nc=256, output_nc=512, dilation=4, norm=norm)]
        model += [ResBlock(input_nc=512, output_nc=512, dilation=4, norm=norm)]
        # level 7
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, dilation=2)]
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, activation='none', dilation=2)]
        # level 8
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm)]
        model += [Conv2dBlock(input_nc=512, output_nc=512, kernel_size=3, padding=1, norm=norm, activation='none')]

        self.model = nn.Sequential(*model)
        self.output_nc = 512

    def forward(self, x):
        return self.model(x)
