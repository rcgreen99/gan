import argparse
import time
from typing import Callable
import math

from PIL import Image
from alive_progress import alive_bar, config_handler
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.optim import Adam
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os

from src.models.resnet_generator import ResNetGenerator
from src.models.resnet_discriminator import ResNetDiscriminator

from src.models.conv_generator2 import ConvGenerator
from src.models.conv_discriminator2 import ConvDiscriminator

# from src.models.conv_generator import ConvGenerator
# from src.models.conv_discriminator import ConvDiscriminator
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
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not found")

    return dataset


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
        raise ValueError(f"Model {model} not found")

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


def get_cosine_schedule_with_warmup(initial_value, warmup_steps, max_steps):
    """
    Creates a schedule with a warmup period followed by cosine decay.

    Args:
        initial_value: Target value after warmup and before decay
        warmup_steps: Number of warmup steps
        max_steps: Total number of steps

    Returns:
        Function that takes a step and returns the scheduled value
    """

    def schedule(step):
        if step < warmup_steps:
            # Linear warmup
            return initial_value * step / warmup_steps

        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(1.0, max_steps - warmup_steps)
        progress = min(1.0, progress)
        return initial_value * 0.5 * (1.0 + math.cos(math.pi * progress))

    return schedule


def get_gamma_schedule(start_value, end_value, cooldown_steps, max_steps):
    """
    Creates a schedule for gamma that starts high and decreases over time.

    Args:
        start_value: Initial high gamma value
        end_value: Final low gamma value
        cooldown_steps: Number of steps to maintain the high value before decay
        max_steps: Total number of steps

    Returns:
        Function that takes a step and returns the scheduled gamma value
    """

    def schedule(step):
        if step < cooldown_steps:
            # Maintain high value during cooldown
            return start_value

        # Cosine decay from high to low after cooldown
        progress = (step - cooldown_steps) / max(1.0, max_steps - cooldown_steps)
        progress = min(1.0, progress)
        # Cosine curve from 0 to pi maps to cosine from 1 to -1
        # We want to go from start_value to end_value
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return end_value + (start_value - end_value) * cosine_factor

    return schedule


def train(
    model: str,
    dataset: datasets.VisionDataset,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    noise_dim: int,
    hidden_dim: int,
    dropout: float,
    gamma_start: float = 150.0,
    gamma_end: float = 10.0,
    warmup_epochs: int = 5,
):
    start_time = time.strftime("%Y%m%d_%H%M%S")
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory for saving images
    os.makedirs(f"logs/{start_time}", exist_ok=True)

    # Get the models
    generator, discriminator = get_models(
        model, dataset, noise_dim, hidden_dim, dropout
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
    save_image(
        sample_batch[:64],
        f"logs/{start_time}/real_samples.png",
        nrow=8,
        normalize=True,
    )

    # Create fixed noise for visualization
    fixed_noise = torch.randn(64, noise_dim, device=device)

    # For loss and gradient accumulation
    adv_training = AdversarialTraining(generator, discriminator)

    # Track losses
    g_losses = []
    d_losses = []
    image_variance = []
    global_step = 0

    # Calculate total iterations for schedulers
    total_steps = len(data_loader) * num_epochs
    warmup_steps = len(data_loader) * warmup_epochs
    gamma_cooldown_steps = (
        warmup_steps  # Using same length as warmup for gamma cooldown
    )

    # Create schedulers for learning rate (warmup then decay) and gamma (high to low)
    lr_schedule = get_cosine_schedule_with_warmup(
        learning_rate, warmup_steps, total_steps
    )
    gamma_schedule = get_gamma_schedule(
        gamma_start, gamma_end, gamma_cooldown_steps, total_steps
    )

    # Log the schedule configuration
    print(f"Learning rate schedule: {learning_rate} with {warmup_epochs} warmup epochs")
    print(
        f"Gamma schedule: {gamma_start} to {gamma_end} with {warmup_epochs} cooldown epochs"
    )

    # Track rates for plotting
    lr_values = []
    gamma_values = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        num_batches = 0

        for i, (real_samples, _) in enumerate(data_loader):
            # Update learning rate and gamma based on current step
            current_lr = lr_schedule(global_step)
            current_gamma = gamma_schedule(global_step)

            # Update optimizer learning rates
            for param_group in g_optimizer.param_groups:
                param_group["lr"] = current_lr
            for param_group in d_optimizer.param_groups:
                param_group["lr"] = current_lr

            # Track values for plotting
            lr_values.append(current_lr)
            gamma_values.append(current_gamma)

            real_samples = real_samples.to(device)

            batch_size = real_samples.size(0)
            noise = torch.randn(batch_size, noise_dim, device=device)

            # Train discriminator
            d_optimizer.zero_grad()
            adv_training.accumulate_discriminator_gradients(
                noise, real_samples, gamma=current_gamma
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

                real_scores = discriminator(real_samples)
                fake_scores = discriminator(fake_samples)

                # Compare real scores to fake scores directly
                real_accuracy = (real_scores > fake_scores.mean()).float().mean().item()
                fake_accuracy = (fake_scores > real_scores.mean()).float().mean().item()

                # Generator loss
                g_loss = -discriminator(fake_samples).mean().item()

                # Image statistics
                var = fake_samples.var().item()

                # Accumulate for epoch average
                epoch_d_loss += d_loss
                epoch_g_loss += g_loss
                num_batches += 1

                # # Print batch statistics
                if i % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(data_loader)}")
                    print(
                        f"  Current lr: {current_lr:.6f}, Current gamma: {current_gamma:.2f}"
                    )
                    print(f"  D(real): {d_real:.4f}, D(fake): {d_fake:.4f}")
                    print(f"  G-loss: {g_loss:.4f}, D-loss: {d_loss:.4f}")
                    print(f"  Generated image variance: {var:.6f}")
                    print(
                        f"  Generated range - Min: {fake_samples.min():.4f}, Max: {fake_samples.max():.4f}"
                    )
                    print(
                        f"  Real accuracy: {real_accuracy:.4f}, Fake accuracy: {fake_accuracy:.4f}"
                    )

            global_step += 1

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
            save_image(
                gen_imgs,
                f"logs/{start_time}/epoch_{epoch+1}.png",
                nrow=8,
                normalize=True,
            )

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

        # Plot losses and learning rates every 10 epochs
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(g_losses, label="Generator")
            plt.plot(d_losses, label="Discriminator")
            plt.legend()
            plt.title("Losses")

            plt.subplot(1, 3, 2)
            plt.plot(image_variance, label="Image Variance")
            plt.legend()
            plt.title("Generated Image Variance")

            plt.subplot(1, 3, 3)
            plt.plot(
                lr_values[:: len(data_loader)], label="Learning Rate"
            )  # Downsample for clarity
            plt.plot(
                gamma_values[:: len(data_loader)], label="Gamma"
            )  # Downsample for clarity
            plt.legend()
            plt.title("Learning Rate & Gamma")

            plt.savefig(f"logs/{start_time}/metrics_epoch_{epoch+1}.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["conv", "mlp", "resnet"], default="mlp"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10", "celeba"], default="mnist"
    )
    parser.add_argument("--batch_size", type=int, default=128)  # 256 worked on celeba
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument(
        "--gamma_start",
        type=float,
        default=150.0,
        help="Initial gradient penalty coefficient",
    )
    parser.add_argument(
        "--gamma_end",
        type=float,
        default=10.0,
        help="Final gradient penalty coefficient",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=5, help="Number of warmup/cooldown epochs"
    )
    args = parser.parse_args()

    # Get the dataset
    dataset = get_dataset(args.dataset)

    # Train the model
    train(
        model=args.model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gamma_start=args.gamma_start,
        gamma_end=args.gamma_end,
        warmup_epochs=args.warmup_epochs,
    )
