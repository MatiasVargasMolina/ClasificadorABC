from typing import Any, List, Dict


def validar_producto(producto: Any) -> List[str]:
    errores = []

    if producto.ventas_30d > 0 and producto.visitas_30d == 0:
        errores.append("inconsistencia: ventas_30d > 0 pero visitas_30d = 0")

    if producto.visitas_30d < producto.ventas_30d:
        errores.append("inconsistencia: visitas_30d menores que ventas_30d")

    return errores


def validar_productos(productos: List[Any]) -> Dict[str, List[Any]]:
    validos = []
    invalidos = []

    for producto in productos:
        errores = validar_producto(producto)

        if errores:
            invalidos.append({
                "publication_id": producto.publication_id,
                "errores": errores
            })
        else:
            validos.append(producto)

    return {
        "validos": validos,
        "invalidos": invalidos
    }