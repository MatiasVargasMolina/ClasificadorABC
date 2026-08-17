from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.ml.core.config import get_production_config
from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import evaluate_internal_metrics


def ejecutar_ss_kmeans(
    X,
    proportions=None,
    metodo_asignacion: MetodoAsignacion = "global",
):
    config = get_production_config(
        proportions=proportions,
    )

    modelo = SSEKMeans(
        config=config,
        metodo_asignacion=metodo_asignacion,
    )

    resultados = modelo.fit_predict(X)

    metricas = evaluate_internal_metrics(
        X,
        resultados["categoria"],
    )

    diagnostico = {
        "configuracion": {
            "proportions": dict(modelo.proportions),
            "max_iter": modelo.max_iter,
            "tol": modelo.tol,
            "n_init": modelo.n_init,
            "random_state": modelo.random_state,
            "shuffle_unlabeled": modelo.shuffle_unlabeled,
            "metodo_asignacion": modelo.metodo_asignacion,
        },
        "metodo_asignacion_utilizado": modelo.metodo_asignacion,
        "capacidades_objetivo": modelo.capacities_,
        "conteos_finales": modelo.counts_,
        "iteraciones": modelo.n_iter_,
        "inertia": modelo.inertia_,
        "convergio": modelo.converged_,
        "motivo_termino": modelo.stop_reason_,
        "corridas_convergentes": modelo.converged_runs_,
        "corridas_descartadas": modelo.discarded_runs_,
        "metricas": metricas,
    }

    return resultados, diagnostico