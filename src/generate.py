import argparse
import os

from matplotlib import pyplot as plt
import torch

from src.models.conv_generator import ConvGenerator


def generate(
    generator_path: str,
    num_samples: int,
    output_path: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the generator and discriminator
    num_channels = 3
    image_dim = 32
    noise_dim = 100
    hidden_dim = 1024
    generator = ConvGenerator(num_channels, image_dim, noise_dim, hidden_dim).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    # Generate noise
    noise = torch.randn(num_samples, 100).to(device)

    # Generate images
    images = generator(noise)

    # Save images TODO: Turn into function in util.py
    # Reverse the normalization form [-1, 1] to [0, 1]
    images = (images + 1) / 2

    # Convert the tensor to a numpy array in the range with shape (3, 32, 32)
    images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)

    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis("off")

    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_path", type=str)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--output_path", type=str, default="generated_images.png")
    args = parser.parse_args()

    print(args.generator_path)
    print(args.num_samples)
    print(args.output_path)

    generate(args.generator_path, args.num_samples, args.output_path)
