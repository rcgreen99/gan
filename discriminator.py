import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    A discriminator that takes an image and outputs a probability of whether it is real or fake.
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
    discriminator = Discriminator(num_channels=3, image_dim=32).to(device)
    x = torch.randn(1, 3, 32, 32).to(device)
    print(discriminator(x).shape)

# import torch
# import torch.nn as nn


# class Discriminator(nn.Module):
#     """
#     A discriminator that takes an image and outputs a probability of whether it is real or fake.
#     """

#     def __init__(self, num_channels: int, image_dim: int):
#         super().__init__()

#         self.num_channels = num_channels
#         self.image_dim = image_dim
#         self.init_hidden_dim = 64

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 self.num_channels,
#                 self.init_hidden_dim,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#             ),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.conv2 = nn.Sequential(
#             nn.Conv2d(
#                 self.init_hidden_dim,
#                 self.init_hidden_dim * 2,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#             ),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(
#                 self.init_hidden_dim * 2,
#                 self.init_hidden_dim * 4,
#                 kernel_size=3,
#                 stride=2,
#                 padding=1,
#             ),
#             nn.LeakyReLU(0.2, inplace=True),
#         )

#         # Linear layer to output a single probability.
#         self.linear1 = nn.Sequential(
#             nn.Linear(self.init_hidden_dim * 4 * 4 * 4, 1),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         batch_size = x.shape[0]
#         # Apply each learned convolutional block sequentially.
#         x = self.conv1(x)
#         # assert x.shape == (1, 64, 16, 16)
#         x = self.conv2(x)
#         # assert x.shape == (1, 128, 8, 8)
#         x = self.conv3(x)
#         # assert x.shape == (1, 256, 4, 4)

#         # Flatten the output tensor.
#         x = x.view(batch_size, self.init_hidden_dim * 4 * 4 * 4)
#         x = self.linear1(x)
#         # assert x.shape == (1, 1)
#         # Apply the sigmoid activation function to the output.
#         x = torch.sigmoid(x)
#         return x


# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     discriminator = Discriminator(num_channels=3, image_dim=32).to(device)
#     x = torch.randn(1, 3, 32, 32).to(device)
#     print(discriminator(x).shape)
