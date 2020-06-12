import torch.nn as nn
from .modules import Flatten, Activation
from torch.nn import functional as F
import torch

# class CustomPad(torch.nn.module):
#   def __init__(self, padding):
#     self.padding = padding

#   def forward(self, x):
#     return F.pad(x. self.padding, mode='replicate')

class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):

        if False:
            # import pdb;pdb.set_trace()
            # hack to fix onnx interpolation shift when using bilinear upsampling
            b, c, w, h = x.shape
            w = int(w)
            h = int(h)
            # print(w,h)

            # 128
            # nw, nh = (20,20)
            # 178 x 100
            # nw, nh = (15,25)
            # 398 x 224
            # nw, nh = (31,53)
            # x = F.interpolate(x, size=(nw,nh), mode='nearest')

            nw = w * self.scale_factor
            nh = h * self.scale_factor
            # nw = 1280
            # nh = 720
            x = F.interpolate(x, size=(nh,nw), mode='bilinear', align_corners=False)

            # shift mask s pixels right to fix interpolation miss
            sh, sw = 4, 4
            x = x[:,:,:-sh,:-sw]
            x = F.pad(input=x, pad=(sh, 0, sw, 0), mode='constant', value=-3)

            return x

        else:
            return F.interpolate(x, scale_factor=int(self.scale_factor), mode='bilinear')

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        # upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        upsampling = Upsample(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = Activation(activation)
        super().__init__(conv2d, upsampling, activation)


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)
