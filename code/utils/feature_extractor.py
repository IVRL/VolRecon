import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn2d import ConvBnReLU, ResidualBlock

class FPN_FeatureExtractor(nn.Module):
    """
    Feature pyramid network
    """
    def __init__(self, out_ch=32):
        super(FPN_FeatureExtractor, self).__init__()

        self.in_planes = 16

        self.out_ch = out_ch

        self.conv1 = ConvBnReLU(3,16)
        self.layer1 = self._make_layer(32, stride=2)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)
        
        self.inner = nn.Conv2d(64, 96, 1, stride=1, padding=0, bias=True)

        # output convolution
        self.output = nn.Conv2d(96, out_ch, 3, stride=1, padding=1)

    def _make_layer(self, dim, stride=1):   
        layer1 = ResidualBlock(self.in_planes, dim, stride=stride)
        layer2 = ResidualBlock(dim, dim)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        
        return nn.Sequential(*layers)

    def forward(self, x):
        fea0 = self.conv1(x)
        fea1 = self.layer1(fea0)
        fea2 = self.layer2(fea1)
        fea3 = self.layer3(fea2)
        intra_feat = F.interpolate(fea3, scale_factor=2, mode="bilinear") + self.inner(fea2)
        x_out = self.output(intra_feat)
        
        return x_out