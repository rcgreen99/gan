import argparse
import os
import time

from alive_progress import alive_bar, config_handler
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets

from src.util import (
    get_models,
    get_dataset,
    save_models,
    save_losses,
    generate_and_save_samples,
)

# Progress bar
config_handler.set_global(spinner="dots_waves", bar="classic", length=40)


def train(
    model: str,
    dataset: datasets.VisionDataset,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    noise_dim: int,
    hidden_dim: int,
    dropout: float,
):
    start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    generator, discriminator = get_models(
        model, dataset, noise_dim, hidden_dim, dropout
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

    gen_losses = []
    d_losses = []
    image_variance = []
    for epoch in range(num_epochs):
        with alive_bar(
            len(train_loader),
            title=f"Epoch {epoch + 1}/{num_epochs}",
            length=50,
        ) as bar:
            epoch_gen_loss = 0
            epoch_d_loss = 0
            num_batches = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                # Move the data to the device
                data = data.to(device)
                current_batch_size = data.shape[0]

                # Define the valid and fake labels
                valid = torch.ones(current_batch_size, 1).to(device) * 0.9
                fake = torch.zeros(current_batch_size, 1).to(device) * 0.1

                # Generate a batch of noise vectors for the generator
                noise_vector = torch.randn(current_batch_size, 100).to(device)

                ##################Train the Generator######################
                optimizer_generator.zero_grad()

                # Generate a batch of fake images
                generated_images = generator(noise_vector)

                # See how well the discriminator thinks the generated images are real
                gen_loss = criterion(discriminator(generated_images), valid)

                # Backpropagate the generator loss
                gen_loss.backward()
                optimizer_generator.step()

                ##################Train the Discriminator######################
                optimizer_discriminator.zero_grad()

                # Get the discriminator's prediction on the real data
                real_pred = discriminator(data)
                real_loss = criterion(real_pred, valid)

                # Get the discriminator's prediction on the fake data
                fake_pred = discriminator(generated_images.detach())
                fake_loss = criterion(fake_pred, fake)

                # Combine the losses
                combined_loss = (real_loss + fake_loss) / 2

                # Add monitoring of discriminator predictions
                real_accuracy = (real_pred > 0.5).float().mean().item()
                fake_accuracy = (fake_pred < 0.5).float().mean().item()

                # Backpropagate the combined loss
                combined_loss.backward()
                optimizer_discriminator.step()

                ############################################################

                epoch_gen_loss += gen_loss.item()
                epoch_d_loss += combined_loss.item()
                num_batches += 1

                # Update the progress bar
                if batch_idx % 10 == 0:
                    bar.text(
                        f"G loss: {gen_loss.item():.4f} \nD loss: {combined_loss.item():.4f} \nD real acc: {real_accuracy:.2f} \nD fake acc: {fake_accuracy:.2f}"
                    )
                bar()  # Update the progress bar for the epoch

            # Save epoch metrics
            avg_g_loss = epoch_gen_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            gen_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)

            # Generate and save sample images
            img_variance = generate_and_save_samples(
                generator, noise_vector, start_time, epoch, avg_g_loss, avg_d_loss
            )
            image_variance.append(img_variance)

            # Save the models every 10 epochs and the losses
            if (epoch + 1) % 10 == 0:
                save_models(generator, discriminator, start_time, epoch)
                save_losses(gen_losses, d_losses, image_variance, start_time, epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, choices=["mlp", "conv", "resnet"], default="mlp"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar10", "celeba"], default="mnist"
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--noise_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
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
    )
