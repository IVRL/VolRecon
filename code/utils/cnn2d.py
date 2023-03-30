import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)

    def forward(self,x):
        return F.relu(self.conv(x), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.bn(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBnReLU(in_planes, planes, 3, stride=stride, pad=1)
        self.conv2 = ConvBn(planes, planes, 3, stride=1, pad=1)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = ConvBn(in_planes, planes, 3, stride=stride, pad=1)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.downsample is not None:
            x = self.downsample(x)
        return self.relu(x+y)