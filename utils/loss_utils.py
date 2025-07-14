#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import torchvision.models as models
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, layers=None, use_vgg=True, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        if use_vgg:
            model = models.vgg16(pretrained=True).features
            # 默认提取 VGG 的这些层
            self.layers = layers if layers else ['relu1_2', 'relu2_2', 'relu3_3']
            self.feature_layers = {
                'relu1_2': 3,
                'relu2_2': 8,
                'relu3_3': 15,
                'relu4_3': 22,
            }
        else:
            model = models.resnet50(pretrained=True)
            # 默认提取 ResNet 的这些层
            self.layers = layers if layers else ['layer1', 'layer2', 'layer3']
            self.feature_layers = {
                'layer1': model.layer1,
                'layer2': model.layer2,
                'layer3': model.layer3,
            }

        self.model = nn.Sequential(*list(model.children())[:self._get_last_layer_index()])

        # 冻结参数
        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input, target):
        input_features = self._extract_features(input)
        target_features = self._extract_features(target)

        loss = 0
        for layer in self.layers:
            loss += torch.nn.functional.mse_loss(input_features[layer], target_features[layer])
        return loss

    def _extract_features(self, x):
        features = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.feature_layers.values():
                features[name] = x
        return features

    def _get_last_layer_index(self):
        if isinstance(self.feature_layers, dict):
            indices = [idx if isinstance(idx, int) else max(idx.keys()) for idx in self.feature_layers.values()]
            return max(indices) + 1
        return len(self.feature_layers)


class ResNetPerceptualLoss(nn.Module):
    def __init__(self, layers=None, requires_grad=False):
        super(ResNetPerceptualLoss, self).__init__()

        model = models.resnet50(pretrained=True)

        self.pre_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layers = layers if layers else ['layer1', 'layer2', 'layer3']
        self.feature_layers = {
            'layer1': model.layer1,
            'layer2': model.layer2,
            'layer3': model.layer3,
            'layer4': model.layer4,
        }

        self.model = nn.ModuleDict({name: self.feature_layers[name] for name in self.layers})

        if not requires_grad:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, input, target):
        input = self.pre_layers(input)
        target = self.pre_layers(target)
        input_features = self._extract_features(input)
        target_features = self._extract_features(target)

        loss = 0
        for layer in self.layers:
            loss += torch.nn.functional.mse_loss(input_features[layer], target_features[layer])
        return loss

    def _extract_features(self, x):
        features = {}
        for name, module in self.model.items():
            x = module(x)
            features[name] = x
        return features



def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

