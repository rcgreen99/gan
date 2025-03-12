import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvGenerator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        image_dim: int,
        noise_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.noise_dim = noise_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        self.leaky_relu_slope = 0.2

        # Since we have 3 layers, we divide by 8 as we will double the dim 3 times (2**3 = 8)
        self.init_size = image_dim // 8

        # Linear layer to project the noise vector into an initial feature map.
        self.linear1 = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim * self.init_size * self.init_size),
        )

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
        )

        # Upsampling blocks
        self.conv1 = UpsampleBlock(self.hidden_dim, self.hidden_dim)
        self.conv2 = UpsampleBlock(self.hidden_dim, self.hidden_dim)
        self.conv3 = UpsampleBlock(self.hidden_dim, self.num_channels)

        # Tanh to scale the output to [-1, 1].
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.linear1(x)
        x = x.view(batch_size, self.hidden_dim, self.init_size, self.init_size)

        x = self.block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # If the output shape is not the same, interpolate the output to the target shape.
        if x.shape != (batch_size, self.num_channels, self.image_dim, self.image_dim):
            x = F.interpolate(x, size=(self.image_dim, self.image_dim))

        return self.tanh(x)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator = ConvGenerator(
        num_channels=3,
        image_dim=36,
        noise_dim=100,
        hidden_dim=1024,
    ).to(device)

    noise = torch.randn(4, 100).to(device)

    generated_image = generator(noise)

    print(generated_image.shape)
