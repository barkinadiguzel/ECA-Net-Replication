import torch.nn as nn
from src.layers.conv_layer import ConvLayer
from src.layers.normalization import batch_norm
from src.layers.activation import relu
from src.attention.eca_layer import ECALayer

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_eca=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channels, out_channels, 3, stride, 1)
        self.bn1 = batch_norm(out_channels)
        self.relu = relu()

        self.conv2 = ConvLayer(out_channels, out_channels, 3, 1, 1)
        self.bn2 = batch_norm(out_channels)

        self.eca = ECALayer(out_channels) if use_eca else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                ConvLayer(in_channels, out_channels, 1, stride),
                batch_norm(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.eca(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)
