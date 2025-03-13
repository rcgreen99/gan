import torch
import torch.nn as nn
from torchvision import datasets, transforms

from src.models.conv_generator import ConvGenerator
from src.models.conv_discriminator import ConvDiscriminator
from src.models.mlp_generator import MLPGenerator
from src.models.mlp_discriminator import MLPDiscriminator
from src.models.resnet_generator import ResNetGenerator
from src.models.resnet_discriminator import ResNetDiscriminator
from src.training.trainer import GANTrainer
from src.training.training_session_arg_parser import TrainingSessionArgParser
from src.training.train_config import TrainConfig


# TODO: Either turn into all functinos or all methods
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


class TrainingSession:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> None:
        dataset = get_dataset(self.config.dataset)
        dataloader = self.get_dataloader(dataset)
        generator, discriminator = self.get_models(dataset)
        optimizer_generator, optimizer_discriminator = self.get_optimizers(
            generator, discriminator
        )
        criterion = self.get_criterion()
        trainer = GANTrainer(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            generator_optimizer=optimizer_generator,
            discriminator_optimizer=optimizer_discriminator,
            criterion=criterion,
            device=self.device,
            config=self.config,
        )
        trainer.train()

    def get_models(
        self, dataset: datasets.VisionDataset
    ) -> tuple[nn.Module, nn.Module]:
        generator, discriminator = get_models(
            self.config.model,
            dataset,
            self.config.noise_dim,
            self.config.hidden_dim,
            self.config.dropout,
        )
        generator.to(self.device)
        discriminator.to(self.device)
        return generator, discriminator

    def get_dataloader(
        self, dataset: datasets.VisionDataset
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def get_optimizers(self, generator: nn.Module, discriminator: nn.Module):
        optimizer_generator = torch.optim.Adam(
            generator.parameters(), lr=self.config.learning_rate
        )
        optimizer_discriminator = torch.optim.Adam(
            discriminator.parameters(), lr=self.config.learning_rate
        )
        return optimizer_generator, optimizer_discriminator

    def get_criterion(self):
        return nn.BCELoss().to(self.device)


if __name__ == "__main__":
    parser = TrainingSessionArgParser()
    config = parser.parse_args_to_config()
    training_session = TrainingSession(config)
    training_session.run()
