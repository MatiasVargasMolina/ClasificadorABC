from app.ml.optimization.optuna_optimizer import optimize_sse_kmeans
from app.services.preprocessing_service import ejecutar_preprocesamiento


class OptimizationService:

    def optimize(self, request):
        resultado_preprocesamiento = ejecutar_preprocesamiento(request)

        if not resultado_preprocesamiento["hay_validos"]:
            return {
                "mensaje": resultado_preprocesamiento["mensaje"],
                "productos_invalidos": (
                    resultado_preprocesamiento["productos_invalidos"]
                ),
                "best_params": None,
                "best_value": None,
                "best_trial": None,
                "best_metrics": None,
                "optimization_config": None,
                "trials": [],
            }

        result = optimize_sse_kmeans(
            X=resultado_preprocesamiento["X_modelo"],
            n_trials=30,
            random_state=42,
            optimization_seeds=(0, 42, 123),
        )

        return {
            "mensaje": "Optimización completada correctamente",
            "productos_validos": len(
                resultado_preprocesamiento["productos_validos"]
            ),
            "productos_invalidos": (
                resultado_preprocesamiento["productos_invalidos"]
            ),
            **result,
        }