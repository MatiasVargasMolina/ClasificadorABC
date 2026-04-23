from app.preprocessing.validator import validar_productos
from app.preprocessing.transformer import preparar_datos_modelo
from app.preprocessing.scaler import ajustar_y_transformar


def ejecutar_clasificacion(data):
    """
    Orquesta el flujo completo de preprocesamiento y deja
    los datos listos para el clasificador.
    """

    # 1. Los productos ya vienen validados estructuralmente
    #    desde RequestInput (Pydantic)
    productos = data.productos

    # 2. Validación de reglas de negocio
    resultado_validacion = validar_productos(productos)
    productos_validos = resultado_validacion["validos"]
    productos_invalidos = resultado_validacion["invalidos"]

    # 3. Si no hay productos válidos, retornar respuesta controlada
    if not productos_validos:
        return {
            "resultados": [],
            "invalidos": productos_invalidos,
            "mensaje": "No hay productos válidos para clasificar"
        }

    # 4. Transformación a DataFrame y extracción de features
    df_transformado, X = preparar_datos_modelo(productos_validos)

    # 5. Escalado de features numéricas
    X_escalado, scaler = ajustar_y_transformar(X)

    # 6. Aquí luego irá tu clasificador
    # categorias = clasificar_productos(X_escalado)

    # De momento devolvemos datos intermedios para probar el flujo
    return {
        "mensaje": "Preprocesamiento ejecutado correctamente",
        "productos_validos": len(productos_validos),
        "productos_invalidos": productos_invalidos,
        "df_transformado": df_transformado.to_dict(orient="records"),
        "X_escalado": X_escalado.to_dict(orient="records")
    }