import time

from alive_progress import alive_bar, config_handler
import torch
import torch.nn as nn

from src.util import generate_and_save_samples, save_losses, save_models
from src.training.train_config import TrainConfig


config_handler.set_global(spinner="dots_waves", bar="classic", length=40)


class GANTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: TrainConfig,
        device: str,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloader = dataloader
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.criterion = criterion
        self.config = config
        self.device = device

    def train(self):
        start_time = time.strftime("%Y-%m-%d_%H-%M-%S")
        gen_losses = []
        d_losses = []
        image_variance = []
        for epoch in range(self.config.num_epochs):
            with alive_bar(
                len(self.dataloader),
                title=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                length=50,
            ) as p_bar:
                epoch_gen_loss, epoch_d_loss, num_batches = self.train_one_epoch(p_bar)

            # Save epoch metrics
            avg_g_loss = epoch_gen_loss / num_batches
            avg_d_loss = epoch_d_loss / num_batches
            gen_losses.append(avg_g_loss)
            d_losses.append(avg_d_loss)

            # Generate and save sample images
            noise_vector = torch.randn(
                self.config.batch_size, self.config.noise_dim
            ).to(self.device)
            img_variance = generate_and_save_samples(
                self.generator,
                noise_vector,
                start_time,
                epoch,
                avg_g_loss,
                avg_d_loss,
            )
            image_variance.append(img_variance)

            # Save the models every 10 epochs and the losses
            if (epoch + 1) % 10 == 0:
                save_models(self.generator, self.discriminator, start_time, epoch)
                save_losses(gen_losses, d_losses, image_variance, start_time, epoch)

    def train_one_epoch(self, p_bar) -> tuple[float, float, int]:
        epoch_gen_loss = 0
        epoch_d_loss = 0
        num_batches = 0
        for batch_idx, (real_images, _) in enumerate(self.dataloader):
            real_images = real_images.to(self.device)
            current_batch_size = real_images.shape[0]

            # Define the valid and fake labels
            valid = torch.ones(current_batch_size, 1).to(self.device) * 0.9
            fake = torch.zeros(current_batch_size, 1).to(self.device)  # * 0.1

            # Add noise to real images for discriminator training
            if self.config.add_noise:
                real_images = real_images + 0.1 * torch.randn_like(real_images)

            # Generate a batch of noise vectors for the generator
            noise_vector = torch.randn(current_batch_size, self.config.noise_dim).to(
                self.device
            )

            ##################Train the Generator######################
            self.generator_optimizer.zero_grad()

            # Generate a batch of fake images
            generated_images = self.generator(noise_vector)

            # See how well the discriminator thinks the generated images are real
            gen_loss = self.criterion(self.discriminator(generated_images), valid)

            # Backpropagate the generator loss
            gen_loss.backward()
            self.generator_optimizer.step()

            ##################Train the Discriminator######################
            self.discriminator_optimizer.zero_grad()

            # Get the discriminator's prediction on the real data
            real_pred = self.discriminator(real_images)
            real_loss = self.criterion(real_pred, valid)

            # Get the discriminator's prediction on the fake data
            fake_pred = self.discriminator(generated_images.detach())
            fake_loss = self.criterion(fake_pred, fake)

            # Combine the losses
            combined_loss = (real_loss + fake_loss) / 2

            # Add monitoring of discriminator predictions
            real_accuracy = (real_pred > 0.5).float().mean().item()
            fake_accuracy = (fake_pred < 0.5).float().mean().item()

            # Backpropagate the combined loss
            combined_loss.backward()
            self.discriminator_optimizer.step()

            ############################################################

            epoch_gen_loss += gen_loss.item()
            epoch_d_loss += combined_loss.item()
            num_batches += 1

            # Update the progress bar
            if batch_idx % 10 == 0:
                p_bar.text(
                    f"G loss: {gen_loss.item():.4f} \nD loss: {combined_loss.item():.4f} \
                    \nD real acc: {real_accuracy:.2f} \nD fake acc: {fake_accuracy:.2f}"
                )
            p_bar()

        return epoch_gen_loss, epoch_d_loss, num_batches
