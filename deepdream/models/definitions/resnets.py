import os
from collections import namedtuple


import torch
from torchvision import models
from torch.hub import download_url_to_file


from utils.constants import *


class ResNet50(torch.nn.Module):
    """Only those layers are exposed which have already proven to work nicely."""

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()

        resnet50 = torch.load('models/resnet_fc.pth', map_location=torch.device('cpu')).eval()

        self.layer_names = ['layer10', 'layer11', 'layer20', 'layer21', 'layer30', 'layer31', 'layer40', 'layer41',
                            'layer42_conv1', 'layer42_bn1', 'layer42_conv2']

        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool

        # 3
        self.layer10 = resnet50.layer1[0]
        self.layer11 = resnet50.layer1[1]

        # 4
        self.layer20 = resnet50.layer2[0]
        self.layer21 = resnet50.layer2[1]

        # 6
        self.layer30 = resnet50.layer3[0]
        self.layer31 = resnet50.layer3[1]

        # 3
        self.layer40 = resnet50.layer4[0]
        self.layer41 = resnet50.layer4[1]

        # Go even deeper into ResNet's BottleNeck module for layer 42
        self.layer42_conv1 = resnet50.layer4[1].conv1
        self.layer42_bn1 = resnet50.layer4[1].bn1
        self.layer42_conv2 = resnet50.layer4[1].conv2
        self.layer42_bn2 = resnet50.layer4[1].bn2
        self.layer42_relu = resnet50.layer4[1].relu

        # Set these to False so that PyTorch won't be including them in it's autograd engine - eating up precious memory
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Feel free to experiment with different layers
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer10(x)
        layer10 = x
        x = self.layer11(x)
        layer11 = x
        layer12 = x
        x = self.layer20(x)
        layer20 = x
        x = self.layer21(x)
        layer21 = x
        layer23 = x
        x = self.layer30(x)
        layer30 = x
        x = self.layer31(x)
        layer31 = x
        layer35 = x
        x = self.layer40(x)
        layer40 = x
        x = self.layer41(x)
        layer41 = x

        layer42_identity = layer41
        x = self.layer42_conv1(x)
        layer420 = x
        x = self.layer42_bn1(x)
        layer421 = x
        x = self.layer42_relu(x)
        layer422 = x
        x = self.layer42_conv2(x)
        layer423 = x
        x = self.layer42_bn2(x)
        layer424 = x
        x = self.layer42_relu(x)
        layer425 = x
        layer427 = x
        x += layer42_identity
        layer428 = x
        x = self.relu(x)
        layer429 = x

        # Feel free to experiment with different layers.
        net_outputs = namedtuple("ResNet50Outputs", self.layer_names)
        out = net_outputs(layer10, layer11, layer20, layer21, layer30, layer31, layer40, layer41,
                            layer420, layer421, layer423)
        return out