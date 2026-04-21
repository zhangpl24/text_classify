from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch import Tensor


@dataclass(frozen=True)
class TextCNNConfig:
    batch_size: int
    embed_dim: int
    num_classes: int
    num_filters: int
    kernel_sizes: tuple[int, ...]


class TextCNNModel(nn.Module):
    def __init__(self, config: TextCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.softmax = nn.Softmax(dim=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=1,
                out_channels=self.config.num_filters,
                kernel_size=(kernel_size, config.embed_dim)
            )]
            for kernel_size in config.kernel_sizes
        )

    def forward(self, in_features: Tensor) -> Tensor:
        """
        in_features:  输入特征，形状为 (batch_size, num_words, embed_dim)。
        out_features: 输出概率，形状为 (batch_size, num_classes)。
        """
        pass