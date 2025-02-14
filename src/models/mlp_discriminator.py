import torch
import torch.nn as nn


class MLPDiscriminator(nn.Module):
    """
    A simple MLP discriminator.
    """

    def __init__(self, num_channels: int, image_dim: int):
        super().__init__()
        self.num_channels = num_channels
        self.image_dim = image_dim
        self.init_hidden_dim = 768

        self.model = nn.Sequential(
            nn.Linear(
                self.num_channels * self.image_dim * self.image_dim,
                self.init_hidden_dim,
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.init_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.model(x)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = MLPDiscriminator(num_channels=3, image_dim=32).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(discriminator(x).shape)
