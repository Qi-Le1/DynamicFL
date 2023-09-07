import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn, apply_norm
from config import cfg

from .utils import (
    Conv2dCosineNorm,
    FFNCosineNorm,
    Conv2dPCCNorm,
    FFNPCCNorm,
)



# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride):
#         super(Bottleneck, self).__init__()
#         self.n1 = nn.BatchNorm2d(in_planes)
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.n2 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.n3 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

#     def forward(self, x):
#         out = F.relu(self.n1(x))
#         shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
#         out = self.conv1(out)
#         out = self.conv2(F.relu(self.n2(out)))
#         out = self.conv3(F.relu(self.n3(out)))
#         out += shortcut
#         return out

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        # Because the Batch Normalization is done over the C dimension, computing statistics on (N, H, W) slices
        # C from an expected input of size (N, C, H, W)
        # self.n1 = nn.BatchNorm2d(in_planes)
        if cfg['norm'] == 'bn':
            self.n1 = nn.BatchNorm2d(in_planes)
        elif cfg['norm'] == 'ln':
            self.n1 = nn.GroupNorm(1, in_planes)
        # else:
        #     raise ValueError('wrong norm')
        # print(f'in_planes: {in_planes}')
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.n2 = nn.BatchNorm2d(planes)
        if cfg['norm'] == 'bn':
            self.n2 = nn.BatchNorm2d(planes)
        elif cfg['norm'] == 'ln':
            self.n2 = nn.GroupNorm(1, planes)
        # else:
        #     raise ValueError('wrong norm')
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

        # if cfg['norm'] == 'cn':
        #     self.conv1 = Conv2dCosineNorm(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #     self.conv2 = Conv2dCosineNorm(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #     if stride != 1 or in_planes != self.expansion * planes:
        #         self.shortcut = Conv2dCosineNorm(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
        # elif cfg['norm'] == 'pccn':
        #     self.conv1 = Conv2dPCCNorm(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #     self.conv2 = Conv2dPCCNorm(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #     if stride != 1 or in_planes != self.expansion * planes:
        #         self.shortcut = Conv2dPCCNorm(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):

        # if cfg['norm'] == 'cn' or cfg['norm'] == 'pccn':
        #     out = F.relu(x)
        # else:
        out = F.relu(self.n1(x))
        # print(f'out1: {out}')
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # print(f'shortcut: {shortcut}')
        out = self.conv1(out)
        # print(f'out2: {out}')
        # if cfg['norm'] == 'cn' or cfg['norm'] == 'pccn':
        #     out = self.conv2(F.relu(out))
        # else:
        out = self.conv2(F.relu(self.n2(out)))
        # print(f'out3: {out}')
        out += shortcut
        return out

class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()
        # model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
        # cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        # if cfg['norm'] == 'cn':
        #     self.conv1 = Conv2dCosineNorm(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        # elif cfg['norm'] == 'pccn':
        #     self.conv1 = Conv2dPCCNorm(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        # self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        if cfg['norm'] == 'bn':
            self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        elif cfg['norm'] == 'ln':
            self.n4 = nn.GroupNorm(1, hidden_size[3] * block.expansion)
        # else:
        #     raise ValueError('wrong norm')
        # print(f'latent_size: {hidden_size[3] * block.expansion}')
        self.linear = nn.Linear(hidden_size[3] * block.expansion, target_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        # [1, 1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f(self, x, start_layer_idx):
        if start_layer_idx == -1:
            return self.linear(x)
        
        # if cfg['norm'] in ['cn', 'pccn']:
        #     norm1, norm2 = None, None
        # else:
        #     norm1, norm2 = self.n1, self.n2

        x = self.conv1(x)
        # print(f'x1: {x} {x.size()}\n')
        x = self.layer1(x)
        # print(f'x2: {x} \n')
        x = self.layer2(x)
        # print(f'x3: {x} \n')
        x = self.layer3(x)
        # print(f'x4: {x} \n')
        x = self.layer4(x)
        # print(f'x5: {x} \n')
        # if cfg['norm'] == 'cn' or cfg['norm'] == 'pccn':
        #     x = F.relu(x)
        # else:
        x = F.relu(self.n4(x))
        # print(f'x6: {x} \n')
        # print(f'x6_dtype: {x.dtype}')
        x = F.adaptive_avg_pool2d(x, 1)
        # print(f'x7: {x} \n')
        x = x.view(x.size(0), -1)
        # print(f'x8: {x} \n')
        x = self.linear(x)
            # print(f'x9: {x} \n')
        # the x is latent vector, and we      
        # calculate predict layer(latent vector)

        return x

    def forward(self, input, start_layer_idx=None):
        if start_layer_idx == -1:
            output = {}
            output['target'] = self.f(input['data'], start_layer_idx)
            return output
        
        output = {}
        output['target'] = self.f(input['data'], start_layer_idx)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output


def resnet9():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet9']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model.apply(init_param)
    return model


def resnet18():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet18']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
    model.apply(init_param)
    return model
