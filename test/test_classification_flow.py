from app.ml.core.config import SSEKMeansConfig
from app.schemas.input_schema import RequestInput
from app.services.clasificacion_service import (
    ejecutar_clasificacion,
)


def crear_productos_sinteticos() -> list[dict]:
    """
    Construye un conjunto pequeño de publicaciones con perfiles
    comerciales diferentes para probar el flujo completo de
    clasificación.
    """
    productos = []

    for index in range(20):
        if index < 4:
            ventas = 20 - index
            visitas = 100 - index * 5
            precio = 9000 + index * 100

        elif index < 10:
            ventas = 3
            visitas = 15 + index
            precio = 30000 + index * 100

        else:
            ventas = 0
            visitas = index - 9
            precio = 7000 + index * 50

        productos.append(
            {
                "publication_id": (
                    f"MLC-{index:02d}"
                ),
                "ventas_30d": ventas,
                "visitas_30d": visitas,
                "precio_actual": precio,
                "stock_actual": 10,
                "en_promocion": False,
            }
        )

    return productos


def test_classification_service_full_flow(
    monkeypatch,
):
    """
    Comprueba el flujo real del servicio de clasificación:

    entrada validada
        -> preprocesamiento
        -> escalamiento
        -> SS-E-KMeans
        -> diagnóstico
        -> respuesta por publicación

    Solo se reemplaza la configuración de producción para mantener
    la prueba rápida y reproducible.
    """
    configuracion_prueba = SSEKMeansConfig(
        max_iter=100,
        tol=1e-6,
        n_init=5,
        random_state=42,
        shuffle_unlabeled=True,
    )

    monkeypatch.setattr(
        (
            "app.services.kmeans_service."
            "get_production_config"
        ),
        lambda proportions=None: (
            configuracion_prueba
        ),
    )

    request = RequestInput(
        productos=crear_productos_sinteticos()
    )

    resultado = ejecutar_clasificacion(
        request
    )

    assert resultado["mensaje"] == (
        "Clasificación ejecutada correctamente"
    )

    assert resultado["productos_validos"] == 20
    assert resultado["productos_invalidos"] == []
    assert len(resultado["resultados"]) == 20

    diagnostico = resultado["diagnostico"]

    assert diagnostico[
        "capacidades_objetivo"
    ] == {
        "A": 4,
        "B": 6,
        "C": 10,
    }

    assert diagnostico[
        "conteos_finales"
    ] == {
        "A": 4,
        "B": 6,
        "C": 10,
    }

    categorias_obtenidas = {
        publicacion["categoria"]
        for publicacion
        in resultado["resultados"]
    }

    assert categorias_obtenidas == {
        "A",
        "B",
        "C",
    }

    publication_ids = {
        publicacion["publication_id"]
        for publicacion
        in resultado["resultados"]
    }

    assert len(publication_ids) == 20

    for publicacion in resultado["resultados"]:
        assert publicacion["categoria"] in {
            "A",
            "B",
            "C",
        }

        assert (
            "etiqueta_abc_opcional"
            not in publicacion
        )

    assert "iteraciones" in diagnostico
    assert diagnostico["iteraciones"] >= 1

    assert "inertia" in diagnostico
    assert diagnostico["inertia"] >= 0

    assert "metricas" in diagnostico