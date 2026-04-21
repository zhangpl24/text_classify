from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from torch.utils.data import Dataset
from gensim.models import KeyedVectors

from torch import Tensor,tensor

@dataclass(frozen=True)
class Example:
    in_features: Tensor
    label: int

def load_word2vec(path: str) -> KeyedVectors:
    w2v = KeyedVectors.load_word2vec_format(path, binary=True)
    return w2v

def load_examples(path: str, w2v: KeyedVectors) -> list[Example]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                label, text = line.split("\t")
                text = text.split(" ")
                text = [w2v[token] for token in text]
                text = tensor(text)
                examples.append(Example(in_features=text, label=int(label)))
    return examples


class TextDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> Example:
        return self._examples[idx]

