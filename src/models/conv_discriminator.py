import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            # nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        image_dim: int,
        num_features: int,
        dropout: float,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_dim = image_dim
        self.num_features = num_features

        self.final_dim = image_dim // 8

        self.conv1 = DownsampleBlock(self.num_channels, self.num_features * 2, dropout)
        self.conv2 = DownsampleBlock(
            self.num_features * 2, self.num_features * 4, dropout
        )
        self.conv3 = DownsampleBlock(
            self.num_features * 4, self.num_features * 8, dropout
        )

        # Linear layer to output a single probability.
        self.linear1 = nn.Sequential(
            nn.Linear(self.num_features * 8 * self.final_dim * self.final_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output tensor
        x = x.view(batch_size, -1)

        # Output a single probability
        x = self.linear1(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = ConvDiscriminator(
        num_channels=3, image_dim=32, num_features=1024, dropout=0.3
    ).to(device)
    x = torch.randn(10, 3, 32, 32).to(device)
    output = discriminator(x)
    print(output.shape)
