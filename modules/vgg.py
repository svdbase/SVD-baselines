# -*- coding: UTF-8 -*-
# !/user/bin/python3
# +++++++++++++++++++++++++++++++++++++++++++++++++++
# @File Name: vgg.py
# @Author: Jiang.QY
# @Mail: qyjiang24@gmail.com
# @Date: 19-9-12
# +++++++++++++++++++++++++++++++++++++++++++++++++++
import torch

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['vgg16', 'vgg16_bn']


model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
}


class VGG(nn.Module):
    __mean=[0.485, 0.456, 0.406]
    __std=[0.229, 0.224, 0.225]

    def __init__(self, features, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        self.mac_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x, normalize=True):
        if normalize:
            for image in x:
                for img_, m, s in zip(image, self.__mean, self.__std):
                    img_.div_(255.).sub_(m).div_(s)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def forward_mac(self, x):
        feature = []
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                x = m(x)
                f = self.mac_pool(x).view(x.size(0), -1)
                feature.append(f)
            if isinstance(m, nn.ReLU) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.MaxPool2d):
                x = m(x)
        feature = torch.cat(feature, dim=1)
        return feature


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def vgg16(pretrained=True, **kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['vgg16'])
    return model


def vgg16_bn(pretrained=True, **kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['vgg16_bn'])
    return model


if __name__ == '__main__':
    model = vgg16()
    input = torch.randn((2, 3, 224, 224))
    output = model(input)
    print(output.size())
    output = model.forward_mac(input)
    print(output.size())

