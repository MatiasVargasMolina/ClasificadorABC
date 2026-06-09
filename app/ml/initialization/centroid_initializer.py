from typing import Dict, Optional
import numpy as np
import pandas as pd

LABELS_ABC = ("A", "B", "C")


def compute_initial_score(X: pd.DataFrame) -> pd.Series:
    col_min = X.min(axis=0)
    col_range = X.max(axis=0) - col_min
    safe_range = col_range.replace(0.0, 1.0)
    X_normalized = (X - col_min) / safe_range
    return X_normalized.sum(axis=1)


def initialize_centroids(
    X: pd.DataFrame,
    seed_labels: pd.Series,
    scores: pd.Series,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    centers: Dict[str, np.ndarray] = {}
    used_indices = set()

    if rng is None:
        rng = np.random.default_rng(42)

    for label in LABELS_ABC:
        mask = seed_labels == label

        if mask.any():
            centers[label] = X.loc[mask].mean(axis=0).to_numpy(dtype=float)
            used_indices.update(X.index[mask].tolist())

    sorted_idx = scores.sort_values(ascending=False).index.tolist()

    def pick_random(candidates):
        if not candidates:
            return sorted_idx[0]
        return rng.choice(candidates)

    if "A" not in centers:
        available = [idx for idx in sorted_idx if idx not in used_indices]
        top_zone = available[: max(1, len(available) // 3)]
        idx_a = pick_random(top_zone)
        centers["A"] = X.loc[idx_a].to_numpy(dtype=float)
        used_indices.add(idx_a)

    if "B" not in centers:
        available = [idx for idx in sorted_idx if idx not in used_indices]
        start = len(available) // 3
        end = 2 * len(available) // 3
        mid_zone = available[start:end] or available
        idx_b = pick_random(mid_zone)
        centers["B"] = X.loc[idx_b].to_numpy(dtype=float)
        used_indices.add(idx_b)

    if "C" not in centers:
        available = [idx for idx in sorted_idx if idx not in used_indices]
        low_zone = available[-max(1, len(available) // 3):] or available
        idx_c = pick_random(low_zone)
        centers["C"] = X.loc[idx_c].to_numpy(dtype=float)

    return (
        pd.DataFrame.from_dict(
            centers,
            orient="index",
            columns=X.columns,
        )
        .loc[list(LABELS_ABC)]
    )