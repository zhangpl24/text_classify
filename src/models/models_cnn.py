from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch import Tensor
import torch


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
        self.linear = nn.Linear(config.num_filters * len(config.kernel_sizes),config.num_classes)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList(
            [nn.Conv2d(
                in_channels=1,
                out_channels=self.config.num_filters,
                kernel_size=(kernel_size, config.embed_dim)
            )
            for kernel_size in config.kernel_sizes
            ]
        )

    def forward(self, in_features: Tensor) -> Tensor:
        """
        in_features:  输入特征，形状为 (batch_size, num_words, embed_dim)。
        out_features: 输出概率，形状为 (batch_size, num_classes)。
        """
        after_max_pool = []
        for i in range(len(self.convs)):
            conv = self.convs[i]
            out = conv(in_features.unsqueeze(1))
            # (batch_size, out_channels, after_conv, 1)

            out = out.squeeze(-1)
            # (batch_size, out_channels, after_conv)

            out = self.relu(out)

            out = out.max(dim=2).values
            # (batch_size, out_channels)
            after_max_pool.append(out)

        after_cat = torch.cat(after_max_pool, dim=1)
        out_features = self.linear(after_cat)
        return out_features
        # (batch_size, last_layer)



