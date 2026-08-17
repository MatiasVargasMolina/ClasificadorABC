from app.ml.assignment.constrained_assignment import MetodoAsignacion
from app.ml.core.config import MIN_OPERATIONAL_SAMPLES
from app.services.preprocessing_service import ejecutar_preprocesamiento
from app.services.kmeans_service import ejecutar_ss_kmeans


def ejecutar_clasificacion(
    data,
    metodo_asignacion: MetodoAsignacion = "global",
):
    resultado_preprocesamiento = ejecutar_preprocesamiento(data)

    if not resultado_preprocesamiento["hay_validos"]:
        return {
            "mensaje": resultado_preprocesamiento["mensaje"],
            "productos_validos": 0,
            "resultados": [],
            "productos_invalidos": (
                resultado_preprocesamiento["productos_invalidos"]
            ),
            "minimo_operacional": MIN_OPERATIONAL_SAMPLES,
        }

    cantidad_validos = len(
        resultado_preprocesamiento["productos_validos"]
    )

    if cantidad_validos < MIN_OPERATIONAL_SAMPLES:
        return {
            "mensaje": (
                "Se requieren al menos "
                f"{MIN_OPERATIONAL_SAMPLES} publicaciones válidas "
                "después de excluir registros inconsistentes. "
                f"Se obtuvieron {cantidad_validos}."
            ),
            "productos_validos": cantidad_validos,
            "productos_invalidos": (
                resultado_preprocesamiento["productos_invalidos"]
            ),
            "minimo_operacional": MIN_OPERATIONAL_SAMPLES,
            "resultados": [],
        }

    resultados_kmeans, diagnostico = ejecutar_ss_kmeans(
        X=resultado_preprocesamiento["X_modelo"],
        metodo_asignacion=metodo_asignacion,
    )

    df_resultados = (
        resultado_preprocesamiento["df_transformado"].copy()
    )

    df_resultados["categoria"] = (
        resultados_kmeans["categoria"].values
    )

    df_resultados["score_inicial"] = (
        resultados_kmeans["score_inicial"].values
    )

    return {
        "mensaje": "Clasificación ejecutada correctamente",
        "productos_validos": cantidad_validos,
        "productos_invalidos": (
            resultado_preprocesamiento["productos_invalidos"]
        ),
        "metodo_asignacion_utilizado": (
            diagnostico["metodo_asignacion_utilizado"]
        ),
        "resultados": df_resultados.to_dict(
            orient="records"
        ),
        "diagnostico": {
            "metodo_asignacion_utilizado": (
                diagnostico["metodo_asignacion_utilizado"]
            ),
            "capacidades_objetivo": (
                diagnostico["capacidades_objetivo"]
            ),
            "conteos_finales": (
                diagnostico["conteos_finales"]
            ),
            "iteraciones": diagnostico["iteraciones"],
            "inertia": diagnostico["inertia"],
            "metricas": diagnostico["metricas"],
        },
    }