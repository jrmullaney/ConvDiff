import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRelu(nn.Module):
    """(convolution => ReLU) """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_relu(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvRelu(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
                in_channels , in_channels // 2,
                kernel_size=2, stride=2)
        self.conv = ConvRelu(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
            )

    def forward(self, x):
        return self.conv(x)