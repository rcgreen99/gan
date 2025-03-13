from torch import nn
import torch
from torch.nn import functional as F


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                # groups=32,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                # groups=32,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                # groups=32,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.downsample(x)
        x = self.block(x)
        return F.relu(x + res)


if __name__ == "__main__":
    block = ResNetBlock(3, 64)
    x = torch.randn(10, 3, 32, 32)
    print(block(x).shape)
