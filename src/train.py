import argparse
import time

from alive_progress import alive_bar, config_handler
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.generator import ConvGenerator
from src.models.discriminator import ConvDiscriminator
from src.models.mlp_generator import MLPGenerator
from src.models.mlp_discriminator import MLPDiscriminator
from src.util import save_generated_images

# Progress bar
config_handler.set_global(spinner="dots_waves", bar="classic", length=40)


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
        return (
            ConvGenerator(num_channels, image_dim, noise_dim, hidden_dim),
            ConvDiscriminator(num_channels, image_dim, hidden_dim, dropout),
        )
    elif model_type == "mlp":
        return (
            MLPGenerator(num_channels, image_dim, noise_dim, hidden_dim),
            MLPDiscriminator(num_channels, image_dim, hidden_dim, dropout),
        )
    else:
        raise ValueError(f"Model type {model_type} not found")


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    generator, discriminator = get_models(
        model_type, dataset, noise_dim, hidden_dim, dropout
    )
    generator.to(device)
    discriminator.to(device)

    # Define train loader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    # Define optimizers
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate
    )

    # Define the loss function
    criterion = nn.BCELoss().to(device)

    # Training loop
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")

    print(
        f"Generator model parameters: {sum(p.numel() for p in generator.parameters())}"
    )
    print(
        f"Discriminator model parameters: {sum(p.numel() for p in discriminator.parameters())}"
    )
    for epoch in range(num_epochs):
        with alive_bar(
            len(train_loader),
            title=f"Epoch {epoch + 1}/{num_epochs}",
            length=50,
        ) as bar:
            for batch_idx, (data, _) in enumerate(train_loader):
                # Move the data to the device
                data = data.to(device)
                current_batch_size = data.shape[0]

                # Define the valid and fake labels using label smoothing
                valid = torch.ones(current_batch_size, 1).to(device) * 0.9
                fake = torch.zeros(current_batch_size, 1).to(device) * 0.1

                # Generate a batch of noise vectors for the generator
                noise_vector = torch.randn(current_batch_size, 100).to(device)

                ### Train the Generator ###
                optimizer_generator.zero_grad()

                # Generate a batch of fake images
                generated_images = generator(noise_vector)

                # See how well the discriminator thinks the generated images are real
                gen_loss = criterion(discriminator(generated_images), valid)

                # Backpropagate the generator loss
                gen_loss.backward()
                optimizer_generator.step()

                ############################################################

                ### Train the Discriminator ###
                optimizer_discriminator.zero_grad()

                # Get the discriminator's prediction on the real data
                real_pred = discriminator(data)
                real_loss = criterion(real_pred, valid)

                # Get the discriminator's prediction on the fake data
                fake_pred = discriminator(generated_images.detach())
                fake_loss = criterion(fake_pred, fake)

                # Combine the losses
                combined_loss = (real_loss + fake_loss) / 2

                # Backpropagate the combined loss
                combined_loss.backward()
                optimizer_discriminator.step()

                # Update the progress bar
                if batch_idx % 10 == 0:
                    bar.text(
                        f"Batch: {batch_idx}/{len(train_loader)} \nG loss: {gen_loss.item():.4f} \nD loss: {combined_loss.item():.4f}"
                    )
                bar()  # Update the progress bar for the epoch

            # Save a sample of the generated images
            output_path = f"logs/{start_time}/generated_image_{epoch}.png"
            save_generated_images(generator, 16, device, output_path)

            # Save generator and discriminator every 10 epochs
            if epoch % 10 == 0 and epoch != 0:
                torch.save(
                    generator.state_dict(), f"logs/{start_time}/generator_{epoch}.pth"
                )
                torch.save(
                    discriminator.state_dict(),
                    f"logs/{start_time}/discriminator_{epoch}.pth",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["conv", "mlp"], default="conv"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10"], default="mnist"
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=1024)
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
