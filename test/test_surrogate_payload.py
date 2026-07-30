from app.ml.explainability.surrogate_payload import (
    FEATURE_COLUMNS,
    build_predict_rows_from_request,
    build_train_rows_from_classification,
)
from app.schemas.input_schema import RequestInput


def test_build_train_rows_uses_current_three_features_and_target():
    classification = {
        "resultados": [
            {
                "publication_id": "MLC-1",
                "ventas_30d": 2,
                "visitas_30d": 10,
                "precio_actual": 9990,
                "stock_actual": 50,
                "en_promocion": 1,
                "categoria": "A",
                "score_inicial": 1.2,
            }
        ]
    }
    rows = build_train_rows_from_classification(classification)
    assert rows == [
        {
            "publication_id": "MLC-1",
            "ventas_30d": 2,
            "visitas_30d": 10,
            "precio_actual": 9990.0,
            "categoria": "A",
        }
    ]
    assert FEATURE_COLUMNS == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]


def test_build_train_rows_returns_empty_list_when_results_are_absent():
    assert build_train_rows_from_classification({}) == []


def test_build_predict_rows_excludes_stock_and_promotion():
    request = RequestInput(
        productos=[
            {
                "publication_id": "MLC-1",
                "ventas_30d": 2,
                "visitas_30d": 10,
                "precio_actual": 9990,
                "stock_actual": 50,
                "en_promocion": True,
            }
        ]
    )
    assert build_predict_rows_from_request(request) == [
        {
            "publication_id": "MLC-1",
            "ventas_30d": 2,
            "visitas_30d": 10,
            "precio_actual": 9990.0,
        }
    ]
