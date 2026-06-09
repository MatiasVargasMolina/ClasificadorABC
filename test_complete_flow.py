#!/usr/bin/env python3
"""
Script de prueba para validar el flujo completo de clasificación.
"""

from app.schemas.input_schema import RequestInput, ProductoInput
from app.services.clasificacion_service import ejecutar_clasificacion


def test_complete_flow():
    """Prueba el flujo completo de preprocesamiento -> kmeans -> resultados"""
    
    # Crear datos de prueba
    productos = [
        ProductoInput(
            publication_id="pub1",
            ventas_30d=100,
            visitas_30d=500,
            precio_actual=50.0,
            stock_actual=10,
            en_promocion=False,
            etiqueta_abc_opcional="A",  # Semilla para clase A
        ),
        ProductoInput(
            publication_id="pub2",
            ventas_30d=50,
            visitas_30d=200,
            precio_actual=30.0,
            stock_actual=5,
            en_promocion=True,
            etiqueta_abc_opcional="B",  # Semilla para clase B
        ),
        ProductoInput(
            publication_id="pub3",
            ventas_30d=10,
            visitas_30d=50,
            precio_actual=10.0,
            stock_actual=50,
            en_promocion=False,
            etiqueta_abc_opcional=None,  # Sin etiqueta
        ),
        ProductoInput(
            publication_id="pub4",
            ventas_30d=200,
            visitas_30d=1000,
            precio_actual=100.0,
            stock_actual=5,
            en_promocion=False,
            etiqueta_abc_opcional=None,  # Sin etiqueta
        ),
        ProductoInput(
            publication_id="pub5",
            ventas_30d=5,
            visitas_30d=20,
            precio_actual=5.0,
            stock_actual=100,
            en_promocion=True,
            etiqueta_abc_opcional=None,  # Sin etiqueta
        ),
    ]
    
    request = RequestInput(productos=productos)
    
    print("=" * 60)
    print("PRUEBA DE FLUJO COMPLETO DE CLASIFICACIÓN")
    print("=" * 60)
    print(f"\n📦 Productos de entrada: {len(productos)}\n")
    
    try:
        resultado = ejecutar_clasificacion(request)
        
        print(f"✅ Clasificación completada exitosamente\n")
        print(f"📊 Resultado:")
        print(f"  - Mensaje: {resultado['mensaje']}")
        print(f"  - Productos válidos: {resultado['productos_validos']}")
        print(f"  - Productos inválidos: {resultado['productos_invalidos']}")
        print(f"\n📈 Diagnóstico:")
        diagnostico = resultado["diagnostico"]
        print(f"  - Capacidades objetivo: {diagnostico['capacidades_objetivo']}")
        print(f"  - Conteos finales: {diagnostico['conteos_finales']}")
        print(f"  - Semillas usadas: {diagnostico['semillas_usadas']}")
        print(f"  - Iteraciones: {diagnostico['iteraciones']}")
        print(f"  - Inertia: {diagnostico['inertia']:.4f}")
        print(f"  - Métricas internas:")
        for metrica, valor in diagnostico["metricas"].items():
            print(f"    • {metrica}: {valor}")
        
        print(f"\n🎯 Resultados de clasificación:")
        for i, resultado_prod in enumerate(resultado["resultados"], 1):
            print(f"  {i}. ID: {resultado_prod['publication_id']}")
            print(f"     Categoría: {resultado_prod['categoria']}")
            print(f"     Score inicial: {resultado_prod['score_inicial']:.4f}")
            print(f"     Es semilla: {resultado_prod['es_semilla']}")
        
        print("\n" + "=" * 60)
        print("✅ PRUEBA COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR durante la clasificación:")
        print(f"  {type(e).__name__}: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_complete_flow()
    exit(0 if success else 1)
