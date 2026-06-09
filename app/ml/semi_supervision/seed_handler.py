from typing import Optional, Sequence, Dict
import pandas as pd

LABELS_ABC = ("A", "B", "C")


def normalize_seed_labels(
    seed_labels: Optional[Sequence[object]],
    index: pd.Index,
) -> pd.Series:
    if seed_labels is None:
        return pd.Series([pd.NA] * len(index), index=index, dtype="object")

    if isinstance(seed_labels, pd.Series):
        seed_series = seed_labels.reindex(index).copy()
    else:
        if len(seed_labels) != len(index):
            raise ValueError("seed_labels debe tener el mismo largo que X.")
        seed_series = pd.Series(list(seed_labels), index=index, dtype="object")

    mapping = {
        "A": "A",
        "B": "B",
        "C": "C",
        "a": "A",
        "b": "B",
        "c": "C",
        0: "A",
        1: "B",
        2: "C",
        "0": "A",
        "1": "B",
        "2": "C",
        None: pd.NA,
        "": pd.NA,
        -1: pd.NA,
        "-1": pd.NA,
    }

    def map_value(value):
        if pd.isna(value):
            return pd.NA

        if value in mapping:
            return mapping[value]

        raise ValueError(
            "seed_labels solo puede contener A/B/C, 0/1/2, None, NaN o -1."
        )

    return seed_series.map(map_value)


def count_seed_labels(seed_labels: pd.Series) -> Dict[str, int]:
    return (
        seed_labels
        .value_counts(dropna=True)
        .reindex(LABELS_ABC, fill_value=0)
        .to_dict()
    )