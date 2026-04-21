from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from torch import nn


@dataclass(frozen=True)
class TextRNNConfig:
    vocab_size: int
    embed_dim: int
    num_classes: int
    dropout: float
    cell: Literal["rnn", "gru", "lstm"]
    hidden_size: int
    num_layers: int
    bidirectional: bool
    pad_id: int


class TextRNNModel(nn.Module):
    def __init__(self, config: TextRNNConfig) -> None:
        super().__init__()
        self.config = config
        raise NotImplementedError

    def forward(self, input_ids: Any, attention_mask: Any | None = None) -> Any:
        raise NotImplementedError
