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

def load_examples(path: str, w2v: KeyedVectors, max_len : int) -> list[Example]:
    examples = []
    unk_vec = [0.0] * w2v.vector_size

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                label, text = line.split("\t")
                text = text.split(" ")
                input = []
                for token in text:
                    if token in w2v:
                        input.append(w2v[token])
                    else:
                        input.append(unk_vec)
                input = input[:max_len]
                if len(input) < max_len:
                    for _ in range(max_len - len(input)):
                        input.append(unk_vec)

                input = tensor(input)
                examples.append(Example(in_features=input, label=int(label)))
    return examples


class TextDataset(Dataset):
    def __init__(self, examples: list[Example]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) :
        return self._examples[idx].in_features, self._examples[idx].label

