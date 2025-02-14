import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A three layer convolutional discriminator.
    Takes an image and outputs a single probability of whether it is real or fake.
    """

    def __init__(self, num_channels: int, image_dim: int):
        super().__init__()
        self.num_channels = num_channels
        self.image_dim = image_dim
        self.init_hidden_dim = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                self.num_channels,
                self.init_hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                self.init_hidden_dim,
                self.init_hidden_dim * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                self.init_hidden_dim * 2,
                self.init_hidden_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Linear layer to output a single probability.
        self.linear1 = nn.Sequential(
            nn.Linear(self.init_hidden_dim * 4 * 4 * 4, 1),
            nn.Sigmoid(),  # Output a single probability.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten the output tensor
        x = x.view(batch_size, self.init_hidden_dim * 4 * 4 * 4)
        x = self.linear1(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = Discriminator(num_channels=3, image_dim=32).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(discriminator(x).shape)
