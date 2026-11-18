import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import initialize_weights

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(ConvBlock, self).__init__()
        if isinstance(padding, tuple):
            self.pad = nn.ZeroPad2d(padding)
            pad_val = 0
        else:
            self.pad = nn.Identity()
            pad_val = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad_val, groups=groups, bias=bias)
        # Hayao clean models use LayerNorm (GroupNorm with 1 group)
        self.ins_norm = nn.GroupNorm(1, out_channels, affine=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.ins_norm(self.conv(self.pad(x))))

class InvertedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2):
        super(InvertedResBlock, self).__init__()
        self.use_residual = (in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)

        self.conv_block = ConvBlock(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_block2 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=True)
        self.conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.GroupNorm(1, out_channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out = self.conv_block2(out)
        out = self.norm(self.conv(out))
        if self.use_residual:
            return residual + out
        else:
            return out

class Generator(nn.Module):
    def __init__(self, dataset=''):
        super(Generator, self).__init__()
        self.name = f'{self.__class__.__name__}_{dataset}'

        self.conv_block1 = nn.Sequential(
            ConvBlock(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=(0, 1, 0, 1), bias=False),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv_block2 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=(0, 1, 0, 1), bias=False),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.res_blocks = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            InvertedResBlock(128, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            InvertedResBlock(256, 256, expand_ratio=2),
            ConvBlock(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv_block3 = nn.Sequential(
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.conv_block4 = nn.Sequential(
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            ConvBlock(64, 32, kernel_size=7, padding=3, stride=1, bias=False),
        )

        self.decode_blocks = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.res_blocks(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv_block3(out)
        out = F.interpolate(out, scale_factor=2, mode="bilinear")
        out = self.conv_block4(out)
        img = self.decode_blocks(out)
        return img
