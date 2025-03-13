from dataclasses import dataclass


@dataclass
class TrainConfig:
    model: str
    dataset: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    noise_dim: int
    hidden_dim: int
    dropout: float
