import torch
import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(
        self,
        num_channels: int,
        image_dim: int,
        noise_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.image_dim = image_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.output_dim = self.num_channels * self.image_dim * self.image_dim

        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        img = self.mlp(x)
        img = img.view(batch_size, self.num_channels, self.image_dim, self.image_dim)
        return img


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dim = 36
    generator = MLPGenerator(num_channels=3, noise_dim=100, image_dim=image_dim).to(
        device
    )
    noise = torch.randn(4, 100).to(device)
    generated_image = generator(noise)
    print(generated_image.shape)
