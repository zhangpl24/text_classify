from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class Checkpoint:
    path: Path

    def ensure_dir(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

