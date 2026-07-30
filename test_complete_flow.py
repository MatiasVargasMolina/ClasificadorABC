from app.ml.core.config import SSEKMeansConfig
from app.schemas.input_schema import RequestInput
from app.services.clasificacion_service import ejecutar_clasificacion


def synthetic_products():
    products = []
    for index in range(20):
        if index < 4:
            sales = 20 - index
            visits = 100 - index * 5
            price = 9000 + index * 100
        elif index < 10:
            sales = 3
            visits = 15 + index
            price = 30000 + index * 100
        else:
            sales = 0
            visits = index - 9
            price = 7000 + index * 50
        products.append(
            {
                "publication_id": f"MLC-{index:02d}",
                "ventas_30d": sales,
                "visitas_30d": visits,
                "precio_actual": price,
                "stock_actual": 10,
                "en_promocion": False,
            }
        )
    return products


def test_complete_classification_flow_without_external_seed_labels(
    monkeypatch,
):
    test_config = SSEKMeansConfig(
        max_iter=100,
        tol=1e-6,
        n_init=5,
        random_state=42,
        shuffle_unlabeled=True,
    )
    monkeypatch.setattr(
        "app.services.kmeans_service.get_production_config",
        lambda proportions=None: test_config,
    )
    request = RequestInput(productos=synthetic_products())
    result = ejecutar_clasificacion(request)
    assert result["mensaje"] == "Clasificación ejecutada correctamente"
    assert result["productos_validos"] == 20
    assert result["productos_invalidos"] == []
    assert len(result["resultados"]) == 20
    assert result["diagnostico"]["capacidades_objetivo"] == {
        "A": 4,
        "B": 6,
        "C": 10,
    }
    assert result["diagnostico"]["conteos_finales"] == {
        "A": 4,
        "B": 6,
        "C": 10,
    }
    assert {row["categoria"] for row in result["resultados"]} == {
        "A",
        "B",
        "C",
    }
    assert all(
        "etiqueta_abc_opcional" not in row
        for row in result["resultados"]
    )
