from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import evaluate_internal_metrics


def ejecutar_ss_kmeans(
    X,
    proportions=None,
):
    modelo = SSEKMeans(
        proportions=proportions or {
            "A": 0.20,
            "B": 0.30,
            "C": 0.50,
        },
        max_iter=300,
        tol=1e-4,
        n_init=5,
        random_state=42,
        shuffle_unlabeled=False,
    )

    resultados = modelo.fit_predict(X)

    metricas = evaluate_internal_metrics(
        X,
        resultados["categoria"],
    )

    diagnostico = {
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