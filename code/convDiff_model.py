import torch.nn as nn
import torch.nn.functional as F

from convDiff_parts import *

class convDiff(nn.Module):
    def __init__(self):
        super(convDiff, self).__init__()
        
        self.inc = ConvRelu(2, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        #self.down4 = Down(512, 1024)
        #self.up4 = Up(1024, 512)
        self.up3 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.up1 = Up(64, 32)
        self.outc = OutConv(32, 1)
        
    def forward(self, x):
        
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up4(x5, x4)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.outc(x)
        
        return out