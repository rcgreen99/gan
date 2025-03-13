import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGenerator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        image_dim: int,
        noise_dim: int,
        num_features: int,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.num_features = num_features

        # Since we have 3 layers, we divide by 8 as we will double the dim 3 times (2**3 = 8)
        self.init_size = image_dim // 8

        self.linear = nn.Sequential(
            nn.Linear(
                self.noise_dim, self.num_features * self.init_size * self.init_size
            ),
        )
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(),
        )

        self.conv1 = UpsampleBlock(self.num_features, self.num_features * 8)
        self.conv2 = UpsampleBlock(self.num_features * 8, self.num_features * 4)
        self.conv3 = UpsampleBlock(self.num_features * 4, self.num_features * 2)

        self.output_conv = nn.Sequential(
            nn.ConvTranspose2d(
                self.num_features * 2, self.num_channels, 4, 1, 0, bias=False
            ),
        )

        # Tanh to scale the output to [-1, 1].
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.linear(x)
        x = x.view(batch_size, self.num_features, self.init_size, self.init_size)
        x = self.block1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output_conv(x)

        # # If the output shape is not the same, interpolate the output to the target shape.
        if x.shape != (batch_size, self.num_channels, self.image_dim, self.image_dim):
            x = F.interpolate(x, size=(self.image_dim, self.image_dim))

        return self.tanh(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = ConvGenerator(
        num_channels=3,
        image_dim=28,
        noise_dim=100,
        num_features=128,
    ).to(device)

    noise = torch.randn(4, 100).to(device)

    generated_image = generator(noise)

    print(generated_image.shape)
