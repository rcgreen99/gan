import torch
import torch.nn as nn


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvDiscriminator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        image_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.image_dim = image_dim
        self.max_hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim

        self.conv1 = DownsampleBlock(self.num_channels, self.hidden_dim // 4, dropout)
        self.conv2 = DownsampleBlock(
            self.hidden_dim // 4, self.hidden_dim // 2, dropout
        )
        self.conv3 = DownsampleBlock(self.hidden_dim // 2, self.hidden_dim, dropout)

        # Linear layer to output a single probability.
        self.linear1 = nn.Sequential(
            nn.Linear(self.max_hidden_dim * 4 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.conv1(x)
        assert x.shape == (batch_size, self.hidden_dim // 4, 16, 16)
        x = self.conv2(x)
        assert x.shape == (batch_size, self.hidden_dim // 2, 8, 8)
        x = self.conv3(x)
        assert x.shape == (batch_size, self.hidden_dim, 4, 4)

        # Flatten the output tensor
        x = x.view(batch_size, self.hidden_dim * 4 * 4)

        # Output a single probability
        x = self.linear1(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = ConvDiscriminator(
        num_channels=3, image_dim=32, hidden_dim=1024, dropout=0.3
    ).to(device)
    x = torch.randn(10, 3, 32, 32).to(device)
    output = discriminator(x)
    print(output.shape)
