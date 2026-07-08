from app.preprocessing.validator import validar_productos
from app.preprocessing.transformer import preparar_datos_modelo
from app.preprocessing.scaler import normalizar_minmax


def ejecutar_preprocesamiento(data):
    """
    Orquesta el flujo completo de preprocesamiento:
    validación de negocio, transformación y normalización.
    """

    productos = data.productos

    resultado_validacion = validar_productos(productos)

    productos_validos = resultado_validacion["validos"]
    productos_invalidos = resultado_validacion["invalidos"]

    if not productos_validos:
        return {
            "hay_validos": False,
            "mensaje": "No hay productos válidos para clasificar",
            "productos_validos": [],
            "productos_invalidos": productos_invalidos,
            "df_transformado": None,
            "X": None,
            "X_modelo": None,
            "normalizacion": None,
        }

    df_transformado, X = preparar_datos_modelo(productos_validos)

    X_modelo, normalizacion = normalizar_minmax(X)

    return {
        "hay_validos": True,
        "mensaje": "Preprocesamiento ejecutado correctamente",
        "productos_validos": productos_validos,
        "productos_invalidos": productos_invalidos,
        "df_transformado": df_transformado,
        "X": X,
        "X_modelo": X_modelo,
        "normalizacion": normalizacion,
    }