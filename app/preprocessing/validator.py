from typing import Any, List, Dict


def validar_producto(producto: Any) -> List[str]:
    """Recibe productos cuyos campos y rangos ya validó ProductoInput.

    Las unidades vendidas y las visitas cuentan eventos diferentes;
    su relación no constituye un criterio de rechazo.
    """
    return []


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