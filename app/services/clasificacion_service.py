from app.services.preprocessing_service import ejecutar_preprocesamiento
from app.services.kmeans_service import ejecutar_ss_kmeans


def ejecutar_clasificacion(data):
    resultado_preprocesamiento = ejecutar_preprocesamiento(data)

    if not resultado_preprocesamiento["hay_validos"]:
        return {
            "mensaje": resultado_preprocesamiento["mensaje"],
            "resultados": [],
            "productos_invalidos": resultado_preprocesamiento["productos_invalidos"],
        }

    # Ejecutar SSEKMeans sin semillas
    resultados_kmeans, diagnostico = ejecutar_ss_kmeans(
        X=resultado_preprocesamiento["X_modelo"],
    )

    # Combinar resultados con datos originales transformados
    df_resultados = resultado_preprocesamiento["df_transformado"].copy()
    df_resultados["categoria"] = resultados_kmeans["categoria"].values
    df_resultados["score_inicial"] = resultados_kmeans["score_inicial"].values

    return {
        "mensaje": "Clasificación ejecutada correctamente",
        "productos_validos": len(resultado_preprocesamiento["productos_validos"]),
        "productos_invalidos": resultado_preprocesamiento["productos_invalidos"],
        "resultados": df_resultados.to_dict(orient="records"),
        "diagnostico": {
            "capacidades_objetivo": diagnostico["capacidades_objetivo"],
            "conteos_finales": diagnostico["conteos_finales"],
            "iteraciones": diagnostico["iteraciones"],
            "inertia": diagnostico["inertia"],
            "metricas": diagnostico["metricas"],
        },
    }