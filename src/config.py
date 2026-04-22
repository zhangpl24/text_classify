from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Paths:
    dataset_dir: str = "./Dataset"
    train_file: str = dataset_dir + "/train.txt"
    valid_file: str = dataset_dir + "/validation.txt"
    test_file: str = dataset_dir + "/test.txt"
    word2vec: str = dataset_dir + "/wiki_word2vec_50.bin"
    


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_epochs: int = 5
    max_len: int = 50
    log_every: int = 50
    


@dataclass(frozen=True)
class ModelConfig:
    model_type: Literal["cnn", "rnn"] = "cnn"
    embed_dim: int = 50
    num_classes: int = 2

    cnn_kernel_sizes: tuple[int, ...] = (3, 4, 5)
    cnn_num_channels: int = 100

    rnn_cell: Literal["rnn", "gru", "lstm"] = "gru"
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 1
    rnn_bidirectional: bool = True


@dataclass(frozen=True)
class AppConfig:
    paths: Paths = Paths()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()

