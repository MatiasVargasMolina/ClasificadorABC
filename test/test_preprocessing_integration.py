from app.schemas.input_schema import RequestInput
from app.preprocessing.validator import validar_productos
from app.preprocessing.transformer import preparar_datos_modelo
from app.preprocessing.scaler import ajustar_y_transformar


def test_flujo_completo_preprocesamiento():
    data = {
        "productos": [
            {
                "publication_id": "MLC123",
                "ventas_30d": 10,
                "visitas_30d": 100,
                "precio_actual": 15990,
                "stock_actual": 5,
                "en_promocion": True
            },
            {
                "publication_id": "MLC124",
                "ventas_30d": 2,
                "visitas_30d": 40,
                "precio_actual": 8990,
                "stock_actual": 10,
                "en_promocion": False
            }
        ]
    }

    request = RequestInput(**data)
    resultado_validacion = validar_productos(request.productos)

    assert len(resultado_validacion["validos"]) == 2
    assert len(resultado_validacion["invalidos"]) == 0

    df_transformado, X = preparar_datos_modelo(resultado_validacion["validos"])
    X_escalado, scaler = ajustar_y_transformar(X)

    assert df_transformado.shape[0] == 2
    assert X.shape[0] == 2
    assert X_escalado.shape[0] == 2
    assert scaler is not None