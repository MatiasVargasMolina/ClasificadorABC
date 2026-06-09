# app/services/optimization_service.py

from app.preprocessing.transformer import preparar_datos_modelo
from app.ml.optimization.optuna_optimizer import optimize_sse_kmeans


class OptimizationService:

    def optimize(self, request):
        df_transformado, X = preparar_datos_modelo(request.productos)

        seed_labels = df_transformado["etiqueta_abc_opcional"].tolist()

        result = optimize_sse_kmeans(
            X=X,
            seed_labels=seed_labels,
            n_trials=50,
            random_state=42,
        )

        return {
            "mensaje": "Optimización completada correctamente",
            **result,
        }