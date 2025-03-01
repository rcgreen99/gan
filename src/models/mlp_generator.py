import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPGenerator(nn.Module):
    """
    A generator that transforms a noise vector into an image via a learned upsampling process.
    """

    def __init__(
        self, num_channels: int, noise_dim: int, image_dim: int, dropout: float = 0.25
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.noise_dim = noise_dim
        self.image_dim = image_dim

        self.hidden_dim = 1024

        # Linear layer to project the noise vector into an initial feature map.
        self.linear1 = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(dropout),
        )
        self.linear2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Dropout(dropout),
        )
        self.linear3 = nn.Sequential(
            nn.Linear(
                self.hidden_dim // 2,
                self.num_channels * self.image_dim * self.image_dim,
            ),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts the noise vector into an image.

        Args:
            x (torch.Tensor): A noise tensor of shape (batch_size, noise_dim).

        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, num_channels, image_dim, image_dim).
        """
        batch_size = x.shape[0]
        x = self.linear1(x)
        # assert x.shape == (batch_size, self.hidden_dim)
        x = self.linear2(x)
        # assert x.shape == (batch_size, self.hidden_dim // 2)
        x = self.linear3(x)
        # assert x.shape == (
        #     batch_size,
        #     self.num_channels * self.image_dim * self.image_dim,
        # )
        x = x.view(batch_size, self.num_channels, self.image_dim, self.image_dim)
        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dim = 36
    generator = MLPGenerator(num_channels=3, noise_dim=100, image_dim=image_dim).to(
        device
    )
    noise = torch.randn(4, 100).to(device)
    generated_image = generator(noise)
    print(generated_image.shape)
