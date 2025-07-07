import torch
from torch import nn
from DCNv4.modules.dcnv4 import DCNv4
import torch.nn.functional as F
from ops_dcnv3.modules import DCNv3_pytorch as DCNv3
from dcnv2 import DeformConv2d
from Scconv import ScConv

class DCNV4(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=k)
        self.dcnv4 = DCNv4(out_channels, kernel_size=3, stride=s, group=g, dilation=d)


    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H*W, C)
        x = self.dcnv4(x)
        x = x.reshape(N, H, W, C).permute(0, 3, 1, 2)
        return x



class rsdcnv4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, g=1):
        super(rsdcnv4, self).__init__()
        self.dcnv4 = DCNv4(in_channels, kernel_size=kernel_size, stride=stride, padding=padding, group=g, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    def forward(self, x):
        shortcut = x
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
        x = self.dcnv4(x)
        x = x.reshape(N, H, W, C).permute(0, 3, 1, 2)
        x = self.bn1(x)
        x = F.relu(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
        x = self.dcnv4(x)
        x = x.reshape(N, H, W, C).permute(0, 3, 1, 2)
        x = self.bn1(x)
        # 调整输入的通道数
        if shortcut.size(1) != x.size(1):
            shortcut = self.conv11(shortcut)  # 使用1x1卷积调整通道数
        x += shortcut  # 残差连接
        x= self.conv11(x)
        return x


class rsscconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(rsscconv, self).__init__()
        self.scconv1 = ScConv(in_channels)
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


    def forward(self, x):
        x = self.scconv1(x)
        x = self.conv11(x)
        return x





if __name__ == "__main__":
    model = rsscconv(16, 32).cuda()
    input = torch.randn(16, 16, 128, 128).cuda()
    output = model(input)
    print(output.shape)