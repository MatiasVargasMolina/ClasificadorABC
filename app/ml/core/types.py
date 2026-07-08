from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


LABELS_ABC = ("A", "B", "C")


@dataclass(frozen=True)
class RunResult:
    labels: pd.Series
    centers: pd.DataFrame
    inertia: float
    objective_history: List[float]
    n_iter: int
    capacities: Dict[str, int]
    counts: Dict[str, int]
    scores: pd.Series