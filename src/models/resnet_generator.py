import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet_block import ResNetBlock


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            ResNetBlock(in_channels, out_channels),
            ResNetBlock(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.res_blocks(x)
        return x


class ResNetGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        in_channels: int,
        out_channels: int,
        num_stages: int,
        channels: int,
        image_dim: int,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stages = num_stages
        self.channels = channels
        self.init_size = image_dim // 2**num_stages
        self.head = nn.Sequential(
            nn.Linear(noise_dim, channels * self.init_size**2),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList(
            [UpsampleBlock(channels, channels) for _ in range(num_stages)]
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = self.head(x)

        x = x.view(batch_size, self.channels, self.init_size, self.init_size)

        for block in self.blocks:
            x = block(x)
        return self.output_conv(x)


if __name__ == "__main__":
    generator = ResNetGenerator(
        noise_dim=100,
        in_channels=3,
        out_channels=3,
        num_stages=4,
        channels=64,
        image_dim=32,
    )
    x = torch.randn(10, 100)
    print(generator(x).shape)
