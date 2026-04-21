from __future__ import annotations

import argparse
from typing import Any, Literal

from numpy import rec
from torch.optim import optimizer

from src.config import AppConfig
from src.data import load_examples, TextDataset, load_word2vec
from src.metrics import accuracy, macro_f1
from src.utils import set_seed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gensim.models import KeyedVectors

from src.models.models_cnn import TextCNNConfig, TextCNNModel
from src.models.models_rnn import TextRNNConfig, TextRNNModel

# Load dataset
def load_dataset(cfg: AppConfig, w2v: KeyedVectors) -> tuple[TextDataset, TextDataset, TextDataset]:
    train_examples = load_examples(cfg.paths.train_file, w2v, max_len=cfg.train.max_len)
    val_examples = load_examples(cfg.paths.valid_file, w2v, max_len=cfg.train.max_len)
    test_examples = load_examples(cfg.paths.test_file, w2v, max_len=cfg.train.max_len)
    return TextDataset(train_examples), TextDataset(val_examples), TextDataset(test_examples)

def build_model(model_type: Literal["cnn", "rnn"], cfg: AppConfig) -> nn.Module:
    if model_type == "cnn":
        model_cfg = TextCNNConfig(
            embed_dim=cfg.model.embed_dim,
            num_classes=cfg.model.num_classes,
            kernel_sizes=cfg.model.cnn_kernel_sizes,
            num_filters=cfg.model.cnn_num_channels,
            batch_size=cfg.train.batch_size
        )
        return TextCNNModel(model_cfg)

    model_cfg = TextRNNConfig(
    )
    return TextRNNModel(model_cfg)


def train_one_epoch(
    model : nn.Module, 
    dataloader : torch.utils.data.DataLoader, 
    cross_entropy_loss, 
    optimizer,
    device 
    ) :
    size = len(dataloader.dataset)    
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device) 
        y = y.to(device)

        pred = model(X)
        loss = cross_entropy_loss(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f},   [{current:>5d}]/{size:>5d}")

@torch.no_grad
def evaluate(model, dataloader, device):
    model.eval()

    correct = 0
    tot = 0
    tot_positive = 0
    predict_positive = 0
    TP = 0
    
    for X,y in dataloader:
        X = X.to(device)
        y = y.to(device)

        out = model(X)
        pred = out.argmax(dim=1)

        correct += (pred == y).sum().item()
        tot += y.size(0)
        TP += ((pred == 1) & (y == 1)).sum().item()
        tot_positive += y.sum().item()
        predict_positive += pred.sum().item()

    acc = correct / tot
    F_score = -1
    if predict_positive != 0 and TP != 0:
        precision = TP / predict_positive 
        recall = TP / tot_positive
        F_score = 2/(1/precision + 1/recall)
    return acc, F_score


def main() -> None:
    cfg = AppConfig()
    set_seed(cfg.train.seed)

    w2v = load_word2vec(cfg.paths.word2vec)
    train_dataset, val_dataset, test_dataset = load_dataset(cfg, w2v)

    train_dataloader = DataLoader(train_dataset, cfg.train.batch_size)
    val_dataloader = DataLoader(val_dataset, cfg.train.batch_size)
    test_dataloader = DataLoader(test_dataset, cfg.train.batch_size)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = build_model("cnn",cfg)
    model = model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=cfg.train.learning_rate)

    for epoch in range(cfg.train.num_epochs):
        print(f"This is the {epoch + 1} times train")
        print("This is the loss")
        train_one_epoch(model, train_dataloader, loss_function, optimizer, device)
        print("This is the valid_test")
        acc, F_score = evaluate(model, val_dataloader, device)
        print(f"The acc is {acc}, and the F_score is {F_score}")



if __name__ == "__main__":
    main()
