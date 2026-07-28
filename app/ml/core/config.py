from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Optional


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
    tol: float = 4.2079886696066345e-06
    n_init: int = 10
    random_state: Optional[int] = 42
    shuffle_unlabeled: bool = True


PRODUCTION_CONFIG = SSEKMeansConfig()


def get_production_config(
    random_state: Optional[int] = None,
    proportions: Optional[Mapping[str, float]] = None,
) -> SSEKMeansConfig:
    """
    Retorna la configuración oficial utilizada por la aplicación.

    Permite reemplazar únicamente la semilla y las proporciones.
    Esto se utiliza en las pruebas de estabilidad para conservar
    todos los parámetros productivos y variar solo random_state.
    """
    cambios = {}

    if random_state is not None:
        cambios["random_state"] = random_state

    if proportions is not None:
        cambios["proportions"] = dict(proportions)

    if not cambios:
        return PRODUCTION_CONFIG

    return replace(
        PRODUCTION_CONFIG,
        **cambios,
    )