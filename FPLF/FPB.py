import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

class FPB(nn.Module):
    def __init__(self, dim, h=80, w=41, fp32fft=True):
        super().__init__()
        self.filter = GlobalFilter(dim, h=h, w=w, fp32fft=fp32fft)
        self.feed_forward = FeedForward(in_channel=dim, out_channel=dim)

    def forward(self, rgb):
        freq = self.feed_forward(self.filter(rgb))
        return rgb + freq


class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=80, w=41, fp32fft=True):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class FeedForward(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channel, out_channel, kernel_size=3, padding=2, dilation=2
            ),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
