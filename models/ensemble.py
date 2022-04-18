from .vgg.vgg import vgg19_bn
from .densenet.densenet import densenet161
from .resnet.resnet import resnet101
from .senet.senet import senet154


from torch import nn
import torch
import os


class EnsembleNet_resnet(nn.Module):
    def __init__(self):
        super(EnsembleNet_resnet, self).__init__()
        self.model_3 = resnet101(pretrained=True).cuda().eval()

    def forward(self, x):
        
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_vgg(nn.Module):
    def __init__(self):
        super(EnsembleNet_vgg, self).__init__()
        self.model_3 = vgg19_bn(pretrained=True).cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_densenet(nn.Module):
    def __init__(self):
        super(EnsembleNet_densenet, self).__init__()
        self.model_3 = densenet161(pretrained=True).cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3


class EnsembleNet_senet(nn.Module):
    def __init__(self):
        super(EnsembleNet_senet, self).__init__()
        self.model_3 = senet154(num_classes=1000, pretrained='imagenet').cuda().eval()

    def forward(self, x):
        output_3 = self.model_3(x)
        return output_3