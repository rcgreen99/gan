import argparse
import time

from alive_progress import alive_bar, config_handler
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.util import save_generated_images

# Progress bar
config_handler.set_global(spinner="dots_waves", bar="classic", length=40)


# TODO: Add parameters for model
def train(
    dataset,
    batch_size,
    num_epochs,
    learning_rate,
):
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    noise_dim = 100
    image_dim = dataset[0][0].shape[1]
    num_channels = dataset[0][0].shape[0]

    # Initialize the generator and discriminator
    generator = Generator(num_channels, noise_dim, image_dim).to(device)
    discriminator = Discriminator(num_channels, image_dim).to(device)

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
    with alive_bar(num_epochs, title="Training Progress", length=50) as bar:
        for epoch in range(num_epochs):
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
            if epoch % 10 == 0:
                torch.save(
                    generator.state_dict(), f"logs/{start_time}/generator_{epoch}.pth"
                )
                torch.save(
                    discriminator.state_dict(),
                    f"logs/{start_time}/discriminator_{epoch}.pth",
                )


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    args = parser.parse_args()

    # Get the dataset
    dataset = get_dataset(args.dataset)

    # Train the model
    train(
        dataset=dataset,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )
