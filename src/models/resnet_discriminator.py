from torch import nn
import torch

from src.models.resnet_block import ResNetBlock


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res_blocks = nn.Sequential(
            ResNetBlock(in_channels, out_channels),
            ResNetBlock(out_channels, out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_blocks(x)
        return self.downsample(x)


class ResNetDiscriminator(nn.Module):
    def __init__(self, in_channels: int, num_stages: int, channels: int):
        super().__init__()
        self.num_stages = num_stages
        self.channels = channels

        # Input layer 1x1 conv
        self.input_conv = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1)

        self.blocks = nn.ModuleList(
            [DownsampleBlock(channels, channels) for _ in range(num_stages)]
        )

        # Output layer
        self.output_conv = nn.Sequential(
            ResNetBlock(channels, channels),
            ResNetBlock(channels, channels),
            nn.Flatten(),
            nn.Linear(channels * 4, 1),
            # nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        return self.output_conv(x)  # Raw logits


if __name__ == "__main__":
    discriminator = ResNetDiscriminator(in_channels=3, num_stages=4, channels=64)
    x = torch.randn(10, 3, 32, 32)
    print(discriminator(x).shape)
