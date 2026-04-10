from typing import Any, List, Tuple


def limpiar_producto(producto: Any) -> Any:
    """
    Aplica reglas de limpieza a un producto individual.
    """

    if getattr(producto, "publication_id", None) is not None:
        producto.publication_id = str(producto.publication_id).strip()

    if getattr(producto, "etiqueta_abc_opcional", None) is not None:
        etiqueta = str(producto.etiqueta_abc_opcional).strip().upper()
        producto.etiqueta_abc_opcional = etiqueta if etiqueta != "" else None

    if getattr(producto, "en_promocion", None) is None:
        producto.en_promocion = False

    return producto


def limpiar_productos(productos: List[Any]) -> List[Any]:
    """
    Aplica limpieza a una lista de productos.
    """
    return [limpiar_producto(producto) for producto in productos]


def eliminar_duplicados(productos: List[Any]) -> Tuple[List[Any], List[str]]:
    """
    Elimina duplicados por publication_id, conservando la primera ocurrencia.
    Retorna productos únicos y lista de IDs duplicados.
    """
    vistos = set()
    unicos = []
    duplicados = []

    for producto in productos:
        publication_id = getattr(producto, "publication_id", None)

        if publication_id not in vistos:
            vistos.add(publication_id)
            unicos.append(producto)
        else:
            duplicados.append(publication_id)

    return unicos, duplicados


def limpiar_datos(productos: List[Any]) -> Tuple[List[Any], List[str]]:
    """
    Ejecuta el flujo completo de limpieza.
    """
    productos_limpios = limpiar_productos(productos)
    productos_unicos, duplicados = eliminar_duplicados(productos_limpios)
    return productos_unicos, duplicados