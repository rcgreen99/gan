import os

from PIL import Image
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from src.models.conv_generator2 import ConvGenerator
from src.models.conv_discriminator2 import ConvDiscriminator
from src.models.mlp_generator import MLPGenerator
from src.models.mlp_discriminator import MLPDiscriminator
from src.models.resnet_generator import ResNetGenerator
from src.models.resnet_discriminator import ResNetDiscriminator


def get_models(
    model: str,
    dataset: datasets.VisionDataset,
    noise_dim: int,
    hidden_dim: int,
    dropout: float,
) -> tuple[nn.Module, nn.Module]:
    num_channels = dataset[0][0].shape[0]
    image_dim = dataset[0][0].shape[1]
    if model == "conv":
        generator = ConvGenerator(num_channels, image_dim, noise_dim, hidden_dim)
        discriminator = ConvDiscriminator(num_channels, image_dim, hidden_dim, dropout)
    elif model == "resnet":
        generator = ResNetGenerator(
            noise_dim,
            num_channels,
            num_channels,
            4,
            hidden_dim,
            image_dim,
        )
        discriminator = ResNetDiscriminator(num_channels, 4, hidden_dim)
    elif model == "mlp":
        generator = MLPGenerator(num_channels, image_dim, noise_dim, hidden_dim)
        discriminator = MLPDiscriminator(num_channels, image_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Model type {model} not found")

    # print the parameters in MB
    print(
        f"Generator parameters: {sum(p.numel() for p in generator.parameters()) / 1e6} MB"
    )
    print(
        f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()) / 1e6} MB"
    )
    # Print model architectures for debugging
    print(f"\nGenerator Architecture:\n{generator}")
    print(f"\nDiscriminator Architecture:\n{discriminator}")
    return generator, discriminator


def get_dataset(dataset_name: str):
    if "mnist" in dataset_name:
        # Load the MNIST dataset
        dataset = datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
            ),
        )
    elif "cifar" in dataset_name:
        # Load the CIFAR10 dataset
        dataset = datasets.CIFAR10(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    elif "celeba" in dataset_name:
        # Load the CelebA dataset
        dataset = datasets.CelebA(
            "data",
            split="train",
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


def save_models(generator, discriminator, start_time, epoch):
    os.makedirs(f"logs/{start_time}", exist_ok=True)
    torch.save(generator.state_dict(), f"logs/{start_time}/generator_{epoch}.pth")
    torch.save(
        discriminator.state_dict(), f"logs/{start_time}/discriminator_{epoch}.pth"
    )


def save_losses(gen_losses, d_losses, image_variance, start_time, epoch):
    # Save a plot of the losses
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gen_losses, label="Generator")
    plt.plot(d_losses, label="Discriminator")
    plt.legend()
    plt.title("Losses")

    plt.subplot(1, 2, 2)
    plt.plot(image_variance, label="Image Variance")
    plt.legend()
    plt.title("Generated Image Variance")

    plt.savefig(f"logs/{start_time}/losses_epoch_{epoch+1}.png")
    plt.close()


def generate_and_save_samples(
    generator, noise_vector, start_time, epoch, avg_g_loss, avg_d_loss
):
    """Generate, save and evaluate sample images from the generator.

    Args:
        generator: The generator model
        noise_vector: Input noise vector for the generator
        start_time: Timestamp for logging directory
        epoch: Current epoch number
        avg_g_loss: Average generator loss for the epoch
        avg_d_loss: Average discriminator loss for the epoch

    Returns:
        float: The variance of the generated images
    """
    with torch.no_grad():
        gen_imgs = generator(noise_vector).cpu()
        img_variance = gen_imgs.var().item()

        # Save a grid of generated images
        os.makedirs(f"logs/{start_time}", exist_ok=True)
        save_image(
            gen_imgs[:64],
            f"logs/{start_time}/epoch_{epoch}.png",
            nrow=8,
            normalize=True,
        )

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average G-loss: {avg_g_loss:.4f}, Average D-loss: {avg_d_loss:.4f}")
        print(f"  Image variance: {img_variance:.6f}")

        # Early detection of mode collapse
        if img_variance < 0.01:
            print("WARNING: Very low image variance detected - possible mode collapse!")
            if epoch > 10:
                print(
                    "Suggestion: Try adjusting learning rates or adding batch normalization"
                )

    return img_variance
