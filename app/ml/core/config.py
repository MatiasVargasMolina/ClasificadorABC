from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


DEFAULT_PROPORTIONS = {
    "A": 0.20,
    "B": 0.30,
    "C": 0.50,
}


@dataclass(frozen=True)
class SSEKMeansConfig:
    proportions: Dict[str, float] = field(
        default_factory=lambda: DEFAULT_PROPORTIONS.copy()
    )
    max_iter: int = 300
    tol: float = 1e-4
    n_init: int = 5
    random_state: Optional[int] = 42
    shuffle_unlabeled: bool = True