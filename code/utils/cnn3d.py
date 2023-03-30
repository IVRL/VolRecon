import numpy as np
import torch
from torch import nn
from einops import (rearrange, reduce, repeat)

class conv2dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        bn = self.BN(self.conv(x))
        return self.relu(bn)


class conv3dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.BN = torch.nn.BatchNorm3d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        bn = self.BN(self.conv(x))
        return self.relu(bn)


class tconv3dBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1):
        super().__init__()
        self.conv = torch.nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)
        self.BN = torch.nn.BatchNorm3d(num_features=out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        bn = self.BN(self.conv(x))
        return self.relu(bn)


class VolumeRegularization(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn3d0 = conv3dBNReLU(16,16,3,1,1)
        self.cnn3d1 = torch.nn.Sequential(
            conv3dBNReLU(16,16,3,2,1),
            conv3dBNReLU(16,16,3,1,1),
        )
        self.cnn3d2 = torch.nn.Sequential(
            conv3dBNReLU(16,32,3,2,1),
            conv3dBNReLU(32,32,3,1,1),
        )
        self.cnn3d3 = torch.nn.Sequential(
            conv3dBNReLU(32,48,3,2,1),
            conv3dBNReLU(48,48,3,1,1),
        )

        self.d_cnn3d1 = tconv3dBNReLU(48,32,3,2,1,1)
        self.d_cnn3d2 = tconv3dBNReLU(32,16,3,2,1,1)
        self.d_cnn3d3 = tconv3dBNReLU(16,16,3,2,1,1)
        self.last = nn.Conv3d(16, 16, 3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.cnn3d0(x)
        x1 = self.cnn3d1(x0)
        x2 = self.cnn3d2(x1)
        x3 = self.cnn3d3(x2)

        x2_0 = self.d_cnn3d1(x3)
        x1_0 = self.d_cnn3d2(x2_0 + x2)
        x0_0 = self.d_cnn3d3(x1_0 + x1)
        return self.last(x0_0 + x0)
        