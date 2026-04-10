from typing import Any, List, Dict


ABC_VALIDAS = {"A", "B", "C"}


def validar_producto(producto: Any) -> List[str]:
    """
    Valida reglas de negocio y consistencia lógica de un producto.
    No reemplaza la validación estructural de Pydantic.
    """
    errores = []

    publication_id = getattr(producto, "publication_id", None)
    ventas_30d = getattr(producto, "ventas_30d", None)
    visitas_30d = getattr(producto, "visitas_30d", None)
    precio_actual = getattr(producto, "precio_actual", None)
    stock_actual = getattr(producto, "stock_actual", None)
    en_promocion = getattr(producto, "en_promocion", None)
    etiqueta_abc_opcional = getattr(producto, "etiqueta_abc_opcional", None)

    # -------------------------
    # Respaldo básico
    # -------------------------
    if publication_id is None or not str(publication_id).strip():
        errores.append("publication_id vacío o inexistente")

    if ventas_30d is None:
        errores.append("ventas_30d es obligatorio")

    if visitas_30d is None:
        errores.append("visitas_30d es obligatorio")

    if precio_actual is None:
        errores.append("precio_actual es obligatorio")

    if stock_actual is None:
        errores.append("stock_actual es obligatorio")

    if errores:
        return errores

    # -------------------------
    # Reglas de consistencia lógica
    # -------------------------
    if ventas_30d > 0 and visitas_30d == 0:
        errores.append("inconsistencia: ventas_30d > 0 pero visitas_30d = 0")

    if visitas_30d < ventas_30d:
        errores.append("inconsistencia: visitas_30d menores que ventas_30d")

    # Esta regla puedes dejarla como error o advertencia según tu criterio
    if precio_actual <= 0:
        errores.append("inconsistencia: precio_actual debe ser mayor que 0")

    if stock_actual < 0:
        errores.append("inconsistencia: stock_actual no puede ser negativo")

    # -------------------------
    # Reglas sobre opcionales
    # -------------------------
    if en_promocion is not None and not isinstance(en_promocion, bool):
        errores.append("en_promocion debe ser booleano")

    if etiqueta_abc_opcional is not None:
        etiqueta_normalizada = str(etiqueta_abc_opcional).strip().upper()
        if etiqueta_normalizada not in ABC_VALIDAS:
            errores.append("etiqueta_abc_opcional debe ser A, B o C")

    return errores


def validar_productos(productos: List[Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Valida una lista de productos.

    Retorna un diccionario con:
    - validos: productos sin errores
    - invalidos: lista con publication_id y errores
    """
    validos = []
    invalidos = []

    for producto in productos:
        errores = validar_producto(producto)

        if errores:
            invalidos.append({
                "publication_id": getattr(producto, "publication_id", None),
                "errores": errores
            })
        else:
            validos.append(producto)

    return {
        "validos": validos,
        "invalidos": invalidos
    }