from app.ml.core.ss_kmeans import SSEKMeans
from app.ml.metrics.clustering_metrics import evaluate_internal_metrics


def ejecutar_ss_kmeans(
    X_escalado,
    seed_labels=None,
    proportions=None,
):
    modelo = SSEKMeans(
        proportions={
            "A": 0.20,
            "B": 0.30,
            "C": 0.50,
        },
        max_iter=188,
        tol=0.007078716425305124,
        n_init=5,
        random_state=42,
        shuffle_unlabeled=False,
    )

    resultados = modelo.fit_predict(
        X_escalado,
        seed_labels=seed_labels,
    )

    metricas = evaluate_internal_metrics(
        X_escalado,
        resultados["categoria"],
    )

    diagnostico = {
        "capacidades_objetivo": modelo.capacities_,
        "conteos_finales": modelo.counts_,
        "semillas_usadas": modelo.seed_counts_,
        "iteraciones": modelo.n_iter_,
        "inertia": modelo.inertia_,
        "metricas": metricas,
    }

    return resultados, diagnostico