from __future__ import annotations

import argparse
from typing import Any, Literal

from .config import AppConfig
from .data import load_examples, TextDataset, load_word2vec
from .metrics import accuracy, macro_f1
from .utils import set_seed

import torch
from gensim.models import KeyedVectors

from .models.models_cnn import TextCNNConfig, TextCNNModel
from .models.models_rnn import TextRNNConfig, TextRNNModel

# Load dataset
def load_dataset(cfg: AppConfig, w2v: KeyedVectors) -> tuple[TextDataset, TextDataset, TextDataset]:
    train_examples = load_examples(cfg.paths.train_file, w2v)
    val_examples = load_examples(cfg.paths.valid_file, w2v)
    test_examples = load_examples(cfg.paths.test_file, w2v)
    return TextDataset(train_examples), TextDataset(val_examples), TextDataset(test_examples)

def build_model(model_type: Literal["cnn", "rnn"], cfg: AppConfig) -> Any:
    if model_type == "cnn":
        model_cfg = TextCNNConfig(
            embed_dim=cfg.model.embed_dim,
            num_classes=cfg.model.num_classes,
            dropout=cfg.model.dropout,
            kernel_sizes=cfg.model.cnn_kernel_sizes,
            num_channels=cfg.model.cnn_num_channels
        )
        return TextCNNModel(model_cfg)

    model_cfg = TextRNNConfig(
        vocab_size=cfg.model.vocab_size,
        embed_dim=cfg.model.embed_dim,
        dropout=cfg.model.dropout,
    )
    return TextRNNModel(model_cfg)


def train_one_epoch(model: Any) -> dict[str, float]:
    pass


def evaluate(model: Any) -> dict[str, float]:
    pass


def main() -> None:
    cfg = AppConfig()
    set_seed(cfg.train.seed)

    w2v = load_word2vec(cfg.paths.word2vec)
    train_dataset, val_dataset, test_dataset = load_dataset(cfg, w2v)


if __name__ == "__main__":
    main()
