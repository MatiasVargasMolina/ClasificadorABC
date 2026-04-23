from app.schemas.input_schema import RequestInput
from app.preprocessing.transformer import (
    transformar_productos,
    transformar_variables,
    obtener_features_modelo,
    preparar_datos_modelo,
)


def crear_productos():
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
    return request.productos


def test_transformar_productos_crea_dataframe():
    productos = crear_productos()
    df = transformar_productos(productos)

    assert df.shape[0] == 2
    assert "publication_id" in df.columns
    assert "ventas_30d" in df.columns


def test_transformar_variables_convierte_en_promocion():
    productos = crear_productos()
    df = transformar_productos(productos)
    df_transformado = transformar_variables(df)

    assert df_transformado["en_promocion"].tolist() == [1, 0]


def test_obtener_features_modelo_excluye_publication_id():
    productos = crear_productos()
    df_transformado, X = preparar_datos_modelo(productos)

    assert "publication_id" not in X.columns
    assert "etiqueta_abc_opcional" not in X.columns
    assert "ventas_30d" in X.columns