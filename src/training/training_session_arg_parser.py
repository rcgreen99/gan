from argparse import ArgumentParser

from src.training.train_config import TrainConfig


class TrainingSessionArgParser(ArgumentParser):
    def __init__(self):
        super().__init__()

        # Add config file option
        self.add_argument(
            "--config", type=str, help="Path to JSON configuration file", default=None
        )

        # Model architecture options
        self.add_argument(
            "--model", type=str, choices=["mlp", "conv", "resnet"], default="mlp"
        )
        self.add_argument(
            "--dataset",
            type=str,
            choices=["mnist", "cifar10", "celeba"],
            default="mnist",
        )
        self.add_argument("--batch_size", type=int, default=128)
        self.add_argument("--num_epochs", type=int, default=100)
        self.add_argument("--learning_rate", type=float, default=2e-4)
        self.add_argument("--noise_dim", type=int, default=100)
        self.add_argument("--hidden_dim", type=int, default=128)
        self.add_argument("--dropout", type=float, default=0.3)
        self.add_argument("--add_noise", type=bool, default=False)
        self.add_argument("--d_beta1", type=float, default=0.9)
        self.add_argument("--d_beta2", type=float, default=0.999)
        self.add_argument("--g_beta1", type=float, default=0.9)
        self.add_argument("--g_beta2", type=float, default=0.999)

    def parse_args_to_config(self) -> TrainConfig:
        """Parse arguments and return a TrainingConfig object."""
        args = self.parse_args()

        # Check if config file was provided
        if args.config is not None:
            # Check if any other args were explicitly provided
            other_args_provided = False
            for arg_name, arg_value in vars(args).items():
                if arg_name != "config" and arg_value != self.get_default(arg_name):
                    other_args_provided = True
                    break

            if other_args_provided:
                raise ValueError(
                    "When using a config file, no other command line arguments should be provided."
                )

            # Load config from file
            return TrainConfig.from_json(args.config)
        else:
            # Create config from command line args
            return TrainConfig(
                model=args.model,
                dataset=args.dataset,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                learning_rate=args.learning_rate,
                noise_dim=args.noise_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                add_noise=args.add_noise,
                d_beta1=args.d_beta1,
                d_beta2=args.d_beta2,
                g_beta1=args.g_beta1,
                g_beta2=args.g_beta2,
            )
