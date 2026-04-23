import json
import time
from pathlib import Path

from app.schemas.input_schema import RequestInput
from app.preprocessing.validator import validar_productos
from app.preprocessing.transformer import preparar_datos_modelo
from app.preprocessing.scaler import ajustar_y_transformar


def test_preprocesamiento_con_json_real():
    """
    Test de integración + rendimiento usando un archivo JSON real.

    Valida:
    - que el pipeline completo funcione
    - que los datos se procesen correctamente
    - mide tiempos por etapa
    """

    # =========================
    # 1. Construir ruta al JSON
    # =========================
    BASE_DIR = Path(__file__).resolve().parent.parent
    ruta_json = BASE_DIR / "data" / "input_request.json"

    print("\nRuta utilizada:", ruta_json)

    # =========================
    # 2. Cargar JSON
    # =========================
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # =========================
    # 3. Medición total
    # =========================
    inicio_total = time.perf_counter()

    # =========================
    # 4. Schema (Pydantic)
    # =========================
    inicio_schema = time.perf_counter()
    request = RequestInput(**data)
    fin_schema = time.perf_counter()

    # =========================
    # 5. Validator
    # =========================
    inicio_validator = time.perf_counter()
    resultado_validacion = validar_productos(request.productos)
    productos_validos = resultado_validacion["validos"]
    productos_invalidos = resultado_validacion["invalidos"]
    fin_validator = time.perf_counter()

    # =========================
    # 6. Transformer
    # =========================
    inicio_transformer = time.perf_counter()
    df_transformado, X = preparar_datos_modelo(productos_validos)
    fin_transformer = time.perf_counter()

    # =========================
    # 7. Scaler
    # =========================
    inicio_scaler = time.perf_counter()
    X_escalado, scaler = ajustar_y_transformar(X)
    fin_scaler = time.perf_counter()

    fin_total = time.perf_counter()

    # =========================
    # 8. Cálculo de tiempos
    # =========================
    tiempo_schema = fin_schema - inicio_schema
    tiempo_validator = fin_validator - inicio_validator
    tiempo_transformer = fin_transformer - inicio_transformer
    tiempo_scaler = fin_scaler - inicio_scaler
    tiempo_total = fin_total - inicio_total

    # =========================
    # 9. Output (visible con -s)
    # =========================
    print("\n===== RESULTADOS DE RENDIMIENTO =====")
    print(f"Productos válidos: {len(productos_validos)}")
    print(f"Productos inválidos: {len(productos_invalidos)}")
    print(f"Schema:       {tiempo_schema:.6f} s")
    print(f"Validator:    {tiempo_validator:.6f} s")
    print(f"Transformer:  {tiempo_transformer:.6f} s")
    print(f"Scaler:       {tiempo_scaler:.6f} s")
    print(f"Total:        {tiempo_total:.6f} s")
    print(f"Shape transformado: {df_transformado.shape}")
    print(f"Shape escalado:     {X_escalado.shape}")

    # =========================
    # 10. Validaciones mínimas
    # =========================
    assert len(productos_validos) > 0, "No hay productos válidos"
    assert df_transformado.shape[0] == len(productos_validos)
    assert X_escalado.shape[0] == len(productos_validos)

    # Opcional: umbral de rendimiento (ajusta según tu PC)
    assert tiempo_total < 5, "El preprocesamiento es demasiado lento"