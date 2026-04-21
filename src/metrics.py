from __future__ import annotations

from typing import Any


def accuracy(y_pred: list[int], y_true: list[int]) -> float:
    raise NotImplementedError


def macro_f1(y_pred: list[int], y_true: list[int], num_classes: int) -> float:
    raise NotImplementedError


def classification_report(
    y_pred: list[int],
    y_true: list[int],
    num_classes: int,
) -> dict[str, Any]:
    raise NotImplementedError

