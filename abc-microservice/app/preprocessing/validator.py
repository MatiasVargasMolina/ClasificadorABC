def validar_producto(producto):
    errores = []

    # publication_id
    if producto.publication_id is None or not str(producto.publication_id).strip():
        errores.append("publication_id vacío o inexistente")

    # ventas_30d
    if producto.ventas_30d is None:
        errores.append("ventas_30d es obligatorio")
    elif not isinstance(producto.ventas_30d, int):
        errores.append("ventas_30d debe ser entero")
    elif producto.ventas_30d < 0:
        errores.append("ventas_30d no puede ser negativo")

    # visitas_30d
    if producto.visitas_30d is None:
        errores.append("visitas_30d es obligatorio")
    elif not isinstance(producto.visitas_30d, int):
        errores.append("visitas_30d debe ser entero")
    elif producto.visitas_30d < 0:
        errores.append("visitas_30d no puede ser negativo")

    # precio_actual
    if producto.precio_actual is None:
        errores.append("precio_actual es obligatorio")
    elif not isinstance(producto.precio_actual, (int, float)):
        errores.append("precio_actual debe ser numérico")
    elif producto.precio_actual <= 0:
        errores.append("precio_actual debe ser mayor que 0")

    # stock_actual
    if producto.stock_actual is None:
        errores.append("stock_actual es obligatorio")
    elif not isinstance(producto.stock_actual, int):
        errores.append("stock_actual debe ser entero")
    elif producto.stock_actual < 0:
        errores.append("stock_actual no puede ser negativo")

    # en_promocion
    if producto.en_promocion is not None and not isinstance(producto.en_promocion, bool):
        errores.append("en_promocion debe ser booleano")

    # etiqueta_abc_opcional
    if producto.etiqueta_abc_opcional is not None:
        if str(producto.etiqueta_abc_opcional).upper() not in {"A", "B", "C"}:
            errores.append("etiqueta_abc_opcional debe ser A, B o C")

    # consistencia lógica
    if producto.ventas_30d > 0 and producto.visitas_30d == 0:
        errores.append("inconsistencia: ventas > 0 pero visitas = 0")


    return errores