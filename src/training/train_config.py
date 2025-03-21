from dataclasses import dataclass
import json


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

    @staticmethod
    def from_json(json_path: str) -> "TrainConfig":
        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return TrainConfig(**config_dict)
