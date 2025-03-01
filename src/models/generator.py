import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
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
        self.leaky_relu_slope = 0.2

        # Since we have 3 layers, we divide by 8 as we will double the dim 3 times (2**3 = 8)
        self.init_size = image_dim // 8

        # Linear layer to project the noise vector into an initial feature map.
        self.linear1 = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim * self.init_size * self.init_size),
        )

        # Upsampling block 1: Upsample from (hidden_dim, init_size, init_size)
        # to (hidden_dim // 2, init_size*2, init_size*2).
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(self.hidden_dim),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            nn.ConvTranspose2d(
                self.hidden_dim,
                self.hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim // 2),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            nn.Dropout2d(dropout),
        )

        # Upsampling block 2: Upsample from (hidden_dim // 2, init_size*2, init_size*2)
        # to (hidden_dim // 4, init_size*4, init_size*4).
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dim // 2,
                self.hidden_dim // 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(self.hidden_dim // 4),
            nn.LeakyReLU(self.leaky_relu_slope, inplace=True),
            nn.Dropout2d(dropout),
        )

        # Upsampling block 3: Upsample from (hidden_dim // 4, init_size*4, init_size*4)
        # to (num_channels, init_size*8, init_size*8) which equals (num_channels, image_dim, image_dim).
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                self.hidden_dim // 4, num_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # Scales the output to [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that converts the noise vector into an image.

        Args:
            x (torch.Tensor): A noise tensor of shape (batch_size, noise_dim).

        Returns:
            torch.Tensor: Generated image tensor of shape (batch_size, num_channels, image_dim, image_dim).
        """
        batch_size = x.size(0)

        x = self.linear1(x)
        x = x.view(batch_size, self.hidden_dim, self.init_size, self.init_size)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # If the output shape is not the same, interpolate the output to the target shape.
        if x.shape != (batch_size, self.num_channels, self.image_dim, self.image_dim):
            x = F.interpolate(x, size=(self.image_dim, self.image_dim))

        return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dim = 36
    generator = Generator(num_channels=3, noise_dim=100, image_dim=image_dim).to(device)
    noise = torch.randn(4, 100).to(device)
    generated_image = generator(noise)
    print(generated_image.shape)
