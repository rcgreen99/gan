import argparse
import time
from typing import Callable

from PIL import Image
from alive_progress import alive_bar, config_handler
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

from src.models.conv_generator2 import ConvGenerator
from src.models.conv_discriminator import ConvDiscriminator
from src.models.mlp_generator import MLPGenerator
from src.models.mlp_discriminator import MLPDiscriminator
from src.util import save_generated_images

# Progress bar
config_handler.set_global(spinner="dots_waves", bar="classic", length=40)


class AdversarialTraining:
    def __init__(
        self, generator: MLPGenerator, discriminator: MLPDiscriminator
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator

    @staticmethod
    def zero_centered_gradient_penalty(samples: torch.Tensor, critics: torch.Tensor):
        (gradient,) = torch.autograd.grad(
            outputs=critics.sum(), inputs=samples, create_graph=True
        )
        return gradient.square().sum([1, 2, 3])

    def accumulate_generator_gradients(
        self,
        noise: torch.Tensor,
        real_samples: torch.Tensor,
        scale: float = 1,
        preprocessor: Callable = lambda x: x,
    ):
        fake_samples = self.generator(noise)
        real_samples = real_samples.detach()

        fake_logits = self.discriminator(preprocessor(fake_samples))
        real_logits = self.discriminator(preprocessor(real_samples))

        relativistic_logits = fake_logits - real_logits
        adversarial_loss = nn.functional.softplus(-relativistic_logits)

        (scale * adversarial_loss.mean()).backward()

        return [x.detach() for x in [adversarial_loss, relativistic_logits]]

    def accumulate_discriminator_gradients(
        self,
        noise: torch.Tensor,
        real_samples: torch.Tensor,
        gamma: float,
        scale: float = 1,
        preprocessor: Callable = lambda x: x,
    ):
        real_samples = real_samples.detach().requires_grad_(True)
        fake_samples = self.generator(noise).detach().requires_grad_(True)

        real_logits = self.discriminator(preprocessor(real_samples))
        fake_logits = self.discriminator(preprocessor(fake_samples))

        r1_penalty = AdversarialTraining.zero_centered_gradient_penalty(
            real_samples, real_logits
        )
        r2_penalty = AdversarialTraining.zero_centered_gradient_penalty(
            fake_samples, fake_logits
        )

        relativistic_logits = real_logits - fake_logits
        adversarial_loss = nn.functional.softplus(-relativistic_logits)

        discriminator_loss = adversarial_loss + (gamma / 2) * (r1_penalty + r2_penalty)
        (scale * discriminator_loss.mean()).backward()

        return [
            x.detach()
            for x in [adversarial_loss, relativistic_logits, r1_penalty, r2_penalty]
        ]


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
                ]
            ),
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


def get_models(
    model_type: str,
    dataset: datasets.VisionDataset,
    noise_dim: int,
    hidden_dim: int,
    dropout: float,
) -> tuple[nn.Module, nn.Module]:
    num_channels = dataset[0][0].shape[0]
    image_dim = dataset[0][0].shape[1]
    if model_type == "conv":
        generator = ConvGenerator(num_channels, image_dim, noise_dim, hidden_dim)
        discriminator = ConvDiscriminator(num_channels, image_dim, hidden_dim, dropout)
    elif model_type == "mlp":
        generator = MLPGenerator(num_channels, image_dim, noise_dim, hidden_dim)
        discriminator = MLPDiscriminator(num_channels, image_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Model type {model_type} not found")

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


def train(
    model_type: str,
    dataset: datasets.VisionDataset,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    noise_dim: int,
    hidden_dim: int,
    dropout: float,
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for saving images
    os.makedirs("images", exist_ok=True)

    # Get the models
    generator, discriminator = get_models(
        model_type, dataset, noise_dim, hidden_dim, dropout
    )
    generator.to(device)
    discriminator.to(device)

    # Optimizers
    g_optimizer = Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    d_optimizer = Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    # Data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    # Check data normalization
    sample_batch = next(iter(data_loader))[0]
    print(
        f"\nData statistics - Min: {sample_batch.min():.4f}, Max: {sample_batch.max():.4f}, Mean: {sample_batch.mean():.4f}"
    )

    # Save some real samples for reference
    save_image(sample_batch[:64], "images/real_samples.png", nrow=8, normalize=True)

    # Normalize data to [-1, 1] if it's not already
    # TODO: Do this in the dataset
    transform_to_minus1_1 = lambda x: (
        (x * 2 - 1) if x.min() >= 0 and x.max() <= 1 else x
    )

    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, noise_dim, device=device)

    # Training
    adv_training = AdversarialTraining(generator, discriminator)

    # Track losses
    g_losses = []
    d_losses = []
    image_variance = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (real_samples, _) in enumerate(data_loader):
            # TODO: Do this in the dataset
            real_samples = transform_to_minus1_1(real_samples).to(device)
            # real_samples = real_samples.to(device)
            batch_size = real_samples.size(0)
            noise = torch.randn(batch_size, noise_dim, device=device)

            # Train discriminator
            d_optimizer.zero_grad()
            adv_training.accumulate_discriminator_gradients(
                noise, real_samples, gamma=10.0
            )
            d_optimizer.step()

            # Train generator
            g_optimizer.zero_grad()
            adv_training.accumulate_generator_gradients(noise, real_samples)
            g_optimizer.step()

            # Calculate and log losses for monitoring
            with torch.no_grad():
                # Discriminator loss components
                fake_samples = generator(noise)
                d_real = discriminator(real_samples).mean().item()
                d_fake = discriminator(fake_samples).mean().item()
                d_loss = -d_real + d_fake

                # Generator loss
                g_loss = -discriminator(fake_samples).mean().item()

                # Image statistics
                var = fake_samples.var().item()

                # Accumulate for epoch average
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                num_batches += 1

                # # Print batch statistics
                # if i % 10 == 0:
                #     print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(data_loader)}")
                #     print(f"  D(real): {d_real:.4f}, D(fake): {d_fake:.4f}")
                #     print(f"  G-loss: {g_loss:.4f}, D-loss: {d_loss:.4f}")
                #     print(f"  Generated image variance: {var:.6f}")
                #     print(
                #         f"  Generated range - Min: {fake_samples.min():.4f}, Max: {fake_samples.max():.4f}"
                #     )

        # Save epoch metrics
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_loss = epoch_d_loss / num_batches
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Generate and save sample images
        with torch.no_grad():
            gen_imgs = generator(fixed_noise).cpu()
            img_variance = gen_imgs.var().item()
            image_variance.append(img_variance)

            # Save a grid of generated images
            save_image(gen_imgs, f"images/epoch_{epoch+1}.png", nrow=8, normalize=True)

            print(f"\nEpoch {epoch+1} Summary:")
            print(
                f"  Average G-loss: {avg_g_loss:.4f}, Average D-loss: {avg_d_loss:.4f}"
            )
            print(f"  Image variance: {img_variance:.6f}")

            # Early detection of mode collapse
            if img_variance < 0.01:
                print(
                    "WARNING: Very low image variance detected - possible mode collapse!"
                )
                if epoch > 10:
                    print(
                        "Suggestion: Try adjusting learning rates or adding batch normalization"
                    )

        # Plot losses every 10 epochs
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(g_losses, label="Generator")
            plt.plot(d_losses, label="Discriminator")
            plt.legend()
            plt.title("Losses")

            plt.subplot(1, 2, 2)
            plt.plot(image_variance, label="Image Variance")
            plt.legend()
            plt.title("Generated Image Variance")

            plt.savefig(f"images/losses_epoch_{epoch+1}.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["conv", "mlp"], default="mlp"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10", "celeba"], default="mnist"
    )
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    # Get the dataset
    dataset = get_dataset(args.dataset)

    # Train the model
    train(
        model_type=args.model_type,
        dataset=dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
