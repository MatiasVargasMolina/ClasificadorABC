from app.preprocessing.validator import validar_productos
from app.preprocessing.transformer import preparar_datos_modelo
from app.preprocessing.scaler import ajustar_y_transformar


def ejecutar_preprocesamiento(data):
    """
    Orquesta el flujo completo de preprocesamiento:
    validación de negocio, transformación y escalado.
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
            "X_escalado": None,
            "scaler": None,
        }

    df_transformado, X = preparar_datos_modelo(productos_validos)

    X_escalado, scaler = ajustar_y_transformar(X)

    return {
        "hay_validos": True,
        "mensaje": "Preprocesamiento ejecutado correctamente",
        "productos_validos": productos_validos,
        "productos_invalidos": productos_invalidos,
        "df_transformado": df_transformado,
        "X": X,
        "X_escalado": X_escalado,
        "scaler": scaler,
    }