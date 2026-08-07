import json
import time
from pathlib import Path

from app.schemas.input_schema import RequestInput
from app.services.preprocessing_service import ejecutar_preprocesamiento


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "input_request.json"


def test_preprocessing_real_dataset_completes_under_generous_limit():
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8-sig"))
    request = RequestInput.model_validate(payload)
    started = time.perf_counter()
    result = ejecutar_preprocesamiento(request)
    elapsed = time.perf_counter() - started
    assert result["hay_validos"] is True
    assert len(result["productos_validos"]) > 0
    assert len(result["df_transformado"]) == len(result["X_modelo"])
    assert result["X_modelo"].columns.tolist() == [
        "ventas_30d",
        "visitas_30d",
        "precio_actual",
    ]
    assert elapsed < 10.0
