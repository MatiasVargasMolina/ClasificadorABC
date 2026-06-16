from __future__ import annotations

"""
Worker AutoSklearn + SHAP
=========================

Este archivo levanta una API FastAPI independiente para entrenar y explicar
un modelo AutoSklearn.

IMPORTANTE:
-----------
Este worker NO es el modelo principal SS-E-KMeans.

Flujo real del sistema:

1. La API principal ejecuta SS-E-KMeans.
2. SS-E-KMeans genera etiquetas A/B/C.
3. Este worker entrena AutoSklearn usando:
      X = variables de productos
      y = categoría A/B/C generada por SS-E-KMeans
4. Luego SHAP explica el modelo AutoSklearn.

Entonces:

    SS-E-KMeans = clasificador principal.
    AutoSklearn = modelo sustituto supervisado.
    SHAP = explicación del modelo sustituto.

Uso actual:
-----------
- En local se explica una muestra pequeña para evitar timeout.
- En el server del profesor se puede aumentar la cantidad de productos
  explicados modificando variables de entorno.

Variables importantes para modificar mañana:
--------------------------------------------

SHAP_BACKGROUND_SIZE:
    Cantidad de filas usadas como referencia SHAP.

SHAP_MAX_EXPLAIN_ROWS:
    Cantidad máxima de productos a explicar.
    En local: 10
    En server: 827 o más

SHAP_NSAMPLES:
    Cantidad de muestras usadas por KernelExplainer.
    Más alto = más lento, pero explicación más estable.

Ejemplo local:
--------------
SHAP_BACKGROUND_SIZE=20
SHAP_MAX_EXPLAIN_ROWS=10
SHAP_NSAMPLES=30

Ejemplo server:
---------------
SHAP_BACKGROUND_SIZE=50
SHAP_MAX_EXPLAIN_ROWS=827
SHAP_NSAMPLES=50

O si el server aguanta:
-----------------------
SHAP_BACKGROUND_SIZE=100
SHAP_MAX_EXPLAIN_ROWS=827
SHAP_NSAMPLES=100
"""

import os
import time
from pathlib import Path
from typing import Any, List

import autosklearn.classification
import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.ml.explainability.shap_formatter import (
    build_global_importance,
    build_local_contributions,
    normalize_shap_values,
)


# =============================================================================
# 1. VARIABLES QUE USA EL MODELO
# =============================================================================
# Estas son las columnas que AutoSklearn usa para aprender.
#
# publication_id NO se usa para entrenar.
# publication_id solo sirve para identificar el producto en la respuesta.
#
# categoria tampoco está acá porque:
# - En /train se usa como variable objetivo y.
# - En /explain no existe, porque el modelo debe predecirla.
# =============================================================================

FEATURE_COLUMNS = [
    "ventas_30d",
    "visitas_30d",
    "precio_actual",
    "stock_actual",
    "en_promocion",
]


# =============================================================================
# 2. RUTAS DE ARTEFACTOS
# =============================================================================
# Los artefactos son archivos guardados después del entrenamiento.
#
# Se guardan 3 cosas:
#
# 1. autosklearn_model.joblib
#       Modelo AutoSklearn entrenado.
#
# 2. autosklearn_background.joblib
#       Muestra de datos usada por SHAP como referencia.
#
# 3. autosklearn_meta.joblib
#       Métricas, clases, columnas y configuración.
#
# En Docker, ARTIFACT_DIR normalmente apunta a:
#
#       /artifacts
#
# y en docker-compose se monta algo como:
#
#       ./artifacts/autosklearn:/artifacts
#
# =============================================================================

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/artifacts"))

MODEL_PATH = ARTIFACT_DIR / "autosklearn_model.joblib"
BACKGROUND_PATH = ARTIFACT_DIR / "autosklearn_background.joblib"
META_PATH = ARTIFACT_DIR / "autosklearn_meta.joblib"


# =============================================================================
# 3. CONFIGURACIÓN SHAP
# =============================================================================
# Estas variables son las más importantes para cuando quieras escalar SHAP.
#
# Se leen desde variables de entorno. Si no existen, usan valores por defecto.
#
# LOCAL:
# ------
# SHAP_BACKGROUND_SIZE=20
# SHAP_MAX_EXPLAIN_ROWS=10
# SHAP_NSAMPLES=30
#
# SERVER PROFESOR:
# ----------------
# Primera prueba:
# SHAP_BACKGROUND_SIZE=50
# SHAP_MAX_EXPLAIN_ROWS=100
# SHAP_NSAMPLES=50
#
# Segunda prueba:
# SHAP_BACKGROUND_SIZE=50
# SHAP_MAX_EXPLAIN_ROWS=827
# SHAP_NSAMPLES=50
#
# Prueba más pesada:
# SHAP_BACKGROUND_SIZE=100
# SHAP_MAX_EXPLAIN_ROWS=827
# SHAP_NSAMPLES=100
#
# NOTA:
# -----
# KernelExplainer es lento.
# Si explicaste 10 productos en ~27 segundos, 827 puede tomar bastante.
# Por eso más adelante conviene hacer un sistema por lotes/jobs.
# =============================================================================

SHAP_BACKGROUND_SIZE = int(os.getenv("SHAP_BACKGROUND_SIZE", "20"))
SHAP_MAX_EXPLAIN_ROWS = int(os.getenv("SHAP_MAX_EXPLAIN_ROWS", "10"))
SHAP_NSAMPLES = int(os.getenv("SHAP_NSAMPLES", "30"))


# =============================================================================
# 4. SCHEMAS PYDANTIC
# =============================================================================
# Estos modelos validan la entrada de la API.
#
# Pydantic revisa tipos y restricciones antes de que el endpoint ejecute lógica.
# =============================================================================


class TrainRow(BaseModel):
    """
    Fila usada para entrenar AutoSklearn.

    Incluye categoria porque /train aprende:
        X -> categoria

    La categoria viene desde SS-E-KMeans.
    """

    publication_id: str

    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)

    # Se usa int para que el modelo reciba 0 o 1.
    # 0 = no está en promoción.
    # 1 = está en promoción.
    en_promocion: int = Field(..., ge=0, le=1)

    # Etiqueta generada por SS-E-KMeans.
    # Ejemplo: "A", "B" o "C".
    categoria: str


class PredictRow(BaseModel):
    """
    Fila usada para explicar/predecir.

    No incluye categoria porque el modelo ya entrenado debe predecirla.
    """

    publication_id: str

    ventas_30d: int = Field(..., ge=0)
    visitas_30d: int = Field(..., ge=0)
    precio_actual: float = Field(..., gt=0)
    stock_actual: int = Field(..., ge=0)
    en_promocion: int = Field(..., ge=0, le=1)


class TrainRequest(BaseModel):
    """
    Body del endpoint /train.

    rows:
        Lista de productos ya clasificados por SS-E-KMeans.

    time_left_for_this_task:
        Tiempo total que AutoSklearn puede usar para buscar modelos.

    per_run_time_limit:
        Tiempo máximo por modelo candidato dentro de AutoSklearn.
    """

    rows: List[TrainRow] = Field(..., min_items=3)

    # Local: 120 sirve.
    # Server: puedes probar 300, 600 o más.
    time_left_for_this_task: int = Field(default=120, ge=30, le=1800)

    # Local: 30 sirve.
    # Server: puedes probar 60, 120, 300.
    per_run_time_limit: int = Field(default=30, ge=10, le=300)


class ExplainRequest(BaseModel):
    """
    Body del endpoint /explain.

    rows:
        Lista de productos que quieres explicar.

    top_n:
        Número de variables principales a mostrar en top_contribuciones.
    """

    rows: List[PredictRow] = Field(..., min_items=1)
    top_n: int = Field(default=5, ge=1, le=10)


# =============================================================================
# 5. CREACIÓN DE LA API FASTAPI
# =============================================================================

app = FastAPI(title="AutoSklearn Worker")


# =============================================================================
# 6. ENDPOINT DE SALUD
# =============================================================================
# Sirve para verificar:
#
# - Si Docker levantó bien.
# - Si existe el modelo entrenado.
# - Si existe el background SHAP.
# - Si existe la metadata.
# - Qué configuración SHAP está activa.
#
# Endpoint:
#     GET /health
# =============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "autosklearn-worker",
        "artifact_dir": str(ARTIFACT_DIR),
        "model_exists": MODEL_PATH.exists(),
        "background_exists": BACKGROUND_PATH.exists(),
        "meta_exists": META_PATH.exists(),
        "shap_config": {
            "background_size": SHAP_BACKGROUND_SIZE,
            "max_explain_rows": SHAP_MAX_EXPLAIN_ROWS,
            "nsamples": SHAP_NSAMPLES,
        },
    }


# =============================================================================
# 7. FUNCIONES AUXILIARES
# =============================================================================


def ensure_artifact_dir() -> None:
    """
    Crea la carpeta de artefactos si no existe.

    Se usa en /train antes de guardar:
    - modelo
    - background
    - metadata
    """

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def rows_to_dataframe(rows: List[Any]) -> pd.DataFrame:
    """
    Convierte una lista de objetos Pydantic a pandas DataFrame.

    Entrada:
        rows = [TrainRow(...), TrainRow(...)]
        o
        rows = [PredictRow(...), PredictRow(...)]

    Salida:
        DataFrame con columnas:
        publication_id, ventas_30d, visitas_30d, precio_actual,
        stock_actual, en_promocion, y opcionalmente categoria.

    Importante:
        en_promocion se fuerza a int porque el modelo espera 0/1.
    """

    df = pd.DataFrame([row.dict() for row in rows])
    df["en_promocion"] = df["en_promocion"].astype(int)
    return df


def load_artifacts():
    """
    Carga los artefactos necesarios para /explain.

    /explain necesita:
    - modelo AutoSklearn
    - background SHAP
    - metadata

    Si alguno falta, significa que debes ejecutar /train primero.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe modelo entrenado. Ejecuta /train primero.")

    if not BACKGROUND_PATH.exists():
        raise FileNotFoundError("No existe background SHAP. Ejecuta /train primero.")

    if not META_PATH.exists():
        raise FileNotFoundError("No existe metadata. Ejecuta /train primero.")

    model = joblib.load(MODEL_PATH)
    background = joblib.load(BACKGROUND_PATH)
    meta = joblib.load(META_PATH)

    return model, background, meta


def to_feature_dataframe(data) -> pd.DataFrame:
    """
    Asegura que la entrada del modelo siempre sea un DataFrame con columnas.

    Problema que resuelve:
    ----------------------
    SHAP KernelExplainer a veces llama al modelo con numpy arrays.
    Pero AutoSklearn fue entrenado usando pandas DataFrame con nombres de columnas.

    Si AutoSklearn recibe numpy array, algunos pipelines pueden fallar.

    Esta función convierte cualquier entrada a:

        DataFrame[FEATURE_COLUMNS]

    Esto evita errores de columnas.
    """

    if isinstance(data, pd.DataFrame):
        return data[FEATURE_COLUMNS].copy()

    return pd.DataFrame(data, columns=FEATURE_COLUMNS)


def build_predict_proba_fn(model):
    """
    Crea una función predict_proba compatible con SHAP.

    SHAP necesita una función del tipo:

        f(X) -> probabilidades

    Por eso se envuelve model.predict_proba().

    Retorna:
        predict_proba_fn
    """

    def predict_proba_fn(data):
        X_input = to_feature_dataframe(data)
        return model.predict_proba(X_input)

    return predict_proba_fn


def build_predict_fn(model):
    """
    Crea una función predict compatible con el resto del código.

    Diferencia:
    -----------
    predict_fn:
        devuelve la clase predicha.
        Ejemplo: ["A", "B", "C"]

    predict_proba_fn:
        devuelve probabilidades.
        Ejemplo: [[0.7, 0.2, 0.1], ...]
    """

    def predict_fn(data):
        X_input = to_feature_dataframe(data)
        return model.predict(X_input)

    return predict_fn


def get_autosklearn_model_info(model) -> dict:
    """
    Extrae información interna del modelo AutoSklearn.

    AutoSklearn no necesariamente entrena un solo modelo.
    Normalmente prueba muchos pipelines y arma un ensamble.

    Esta función intenta extraer:

    - automl_class:
        Nombre de la clase del modelo.

    - sprint_statistics:
        Resumen del proceso de búsqueda.

    - leaderboard:
        Ranking de modelos probados.

    - models_with_weights:
        Modelos que quedaron en el ensamble y sus pesos.

    Se usan try/except porque algunas funciones pueden fallar
    dependiendo del estado interno de AutoSklearn.
    """

    info = {
        "automl_class": model.__class__.__name__,
        "sprint_statistics": None,
        "leaderboard": None,
        "models_with_weights": None,
    }

    try:
        info["sprint_statistics"] = str(model.sprint_statistics())
    except Exception as exc:
        info["sprint_statistics"] = f"No disponible: {exc}"

    try:
        leaderboard = model.leaderboard()
        info["leaderboard"] = leaderboard.to_dict(orient="records")
    except Exception as exc:
        info["leaderboard"] = f"No disponible: {exc}"

    try:
        info["models_with_weights"] = str(model.show_models())
    except Exception as exc:
        info["models_with_weights"] = f"No disponible: {exc}"

    return info


# =============================================================================
# 8. ENDPOINT /train
# =============================================================================
# Este endpoint entrena el modelo AutoSklearn.
#
# Entrada:
#     productos con categoria A/B/C.
#
# Esa categoria NO viene de AutoSklearn.
# Viene desde SS-E-KMeans.
#
# Objetivo:
#     entrenar un modelo supervisado que imite a SS-E-KMeans.
#
# Esto se llama modelo sustituto o surrogate model.
#
# Endpoint:
#     POST /train
# =============================================================================

@app.post("/train")
def train(request: TrainRequest):
    """
    Entrena AutoSklearn usando productos ya etiquetados por SS-E-KMeans.

    Flujo:
    1. Crea carpeta de artefactos.
    2. Convierte rows a DataFrame.
    3. Separa X e y.
    4. Divide train/test.
    5. Entrena AutoSklearn.
    6. Evalúa fidelidad.
    7. Crea background SHAP.
    8. Guarda modelo, background y metadata.
    """

    # -------------------------------------------------------------------------
    # 1. Asegurar carpeta donde se guardarán modelo/background/meta.
    # -------------------------------------------------------------------------
    ensure_artifact_dir()

    # -------------------------------------------------------------------------
    # 2. Convertir JSON/Pydantic a DataFrame.
    # -------------------------------------------------------------------------
    df = rows_to_dataframe(request.rows)

    # -------------------------------------------------------------------------
    # 3. Separar variables predictoras X y objetivo y.
    #
    # X:
    #   Variables del producto.
    #
    # y:
    #   Categoría A/B/C generada por SS-E-KMeans.
    # -------------------------------------------------------------------------
    X = df[FEATURE_COLUMNS].copy()
    y = df["categoria"].astype(str).copy()

    # -------------------------------------------------------------------------
    # 4. Validar que existan al menos 2 clases distintas.
    #
    # Un clasificador no puede entrenar si todos los datos son de una sola clase.
    # -------------------------------------------------------------------------
    if y.nunique() < 2:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 2 clases distintas para entrenar AutoSklearn.",
        )

    # -------------------------------------------------------------------------
    # 5. Preparar estratificación.
    #
    # Si cada clase tiene al menos 2 ejemplos, usamos stratify para que
    # train y test mantengan proporciones similares de A/B/C.
    #
    # Si alguna clase tiene menos de 2 ejemplos, no se puede estratificar.
    # -------------------------------------------------------------------------
    stratify = y if y.value_counts().min() >= 2 else None

    # -------------------------------------------------------------------------
    # 6. Dividir dataset en entrenamiento y prueba.
    #
    # 80% train
    # 20% test
    #
    # Si falla por estratificación, se reintenta sin stratify.
    # -------------------------------------------------------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=None,
        )

    # -------------------------------------------------------------------------
    # 7. Medir tiempo de entrenamiento.
    # -------------------------------------------------------------------------
    start = time.time()

    # -------------------------------------------------------------------------
    # 8. Crear AutoSklearnClassifier.
    #
    # time_left_for_this_task:
    #   Tiempo total de búsqueda.
    #
    # per_run_time_limit:
    #   Tiempo máximo por pipeline/modelo candidato.
    #
    # memory_limit:
    #   Memoria máxima en MB.
    #
    # seed:
    #   Reproducibilidad.
    #
    # n_jobs=-1:
    #   Usar todos los núcleos disponibles.
    #
    # MODIFICAR EN SERVER:
    # --------------------
    # En el server puedes aumentar los tiempos desde el request:
    #
    # {
    #   "time_left_for_this_task": 600,
    #   "per_run_time_limit": 120
    # }
    # -------------------------------------------------------------------------
    model = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=request.time_left_for_this_task,
        per_run_time_limit=request.per_run_time_limit,
        memory_limit=4096,
        seed=42,
        n_jobs=-1,
    )

    # -------------------------------------------------------------------------
    # 9. Entrenar el surrogate.
    #
    # Aprende:
    #     X_train -> y_train
    #
    # Donde y_train son categorías de SS-E-KMeans.
    # -------------------------------------------------------------------------
    model.fit(X_train, y_train)

    # -------------------------------------------------------------------------
    # 10. Evaluar el modelo en test.
    #
    # accuracy y macro_f1 miden fidelidad respecto a SS-E-KMeans.
    # -------------------------------------------------------------------------
    predict_fn = build_predict_fn(model)
    y_pred = predict_fn(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_total": int(len(X)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classes": sorted(y.unique().tolist()),
        "training_seconds": round(time.time() - start, 3),
    }

    # -------------------------------------------------------------------------
    # 11. Extraer información interna de AutoSklearn.
    # -------------------------------------------------------------------------
    model_info = get_autosklearn_model_info(model)

    # -------------------------------------------------------------------------
    # 12. Crear background SHAP.
    #
    # El background es la referencia contra la que SHAP compara cada producto.
    #
    # Más background:
    #   + más representativo
    #   - más lento
    #
    # Menos background:
    #   + más rápido
    #   - menos estable
    #
    # MODIFICAR EN SERVER:
    # --------------------
    # Cambiar SHAP_BACKGROUND_SIZE en docker-compose:
    #
    # SHAP_BACKGROUND_SIZE=50
    # o
    # SHAP_BACKGROUND_SIZE=100
    # -------------------------------------------------------------------------
    background = X_train.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(X_train)),
        random_state=42,
    ).copy()

    # -------------------------------------------------------------------------
    # 13. Guardar metadata útil para /explain y para auditoría.
    # -------------------------------------------------------------------------
    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "classes": sorted(y.unique().tolist()),
        "model_info": model_info,
        "shap_config": {
            "background_size": int(len(background)),
            "max_explain_rows": SHAP_MAX_EXPLAIN_ROWS,
            "nsamples": SHAP_NSAMPLES,
        },
    }

    # -------------------------------------------------------------------------
    # 14. Persistir artefactos en disco.
    #
    # Después de esto, /explain puede cargar el modelo sin reentrenar.
    # -------------------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(background, BACKGROUND_PATH)
    joblib.dump(meta, META_PATH)

    # -------------------------------------------------------------------------
    # 15. Respuesta final de entrenamiento.
    # -------------------------------------------------------------------------
    return {
        "mensaje": "AutoSklearn entrenado correctamente",
        "metrics": metrics,
        "model_info": model_info,
        "shap_config": meta["shap_config"],
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "background_path": str(BACKGROUND_PATH),
            "meta_path": str(META_PATH),
        },
    }


# =============================================================================
# 9. ENDPOINT /explain
# =============================================================================
# Este endpoint genera explicaciones SHAP.
#
# Entrada:
#     productos sin categoria.
#
# Flujo:
#     1. Carga modelo/background/meta.
#     2. Recorta cantidad de productos según SHAP_MAX_EXPLAIN_ROWS.
#     3. Predice categoría A/B/C.
#     4. Calcula probabilidades.
#     5. Ejecuta SHAP KernelExplainer.
#     6. Genera explicaciones locales por producto.
#     7. Genera importancia global de los productos explicados.
#
# NOTA IMPORTANTE:
# ----------------
# Si SHAP_MAX_EXPLAIN_ROWS=10, la importancia_global es sobre 10 productos.
# Si SHAP_MAX_EXPLAIN_ROWS=827, la importancia_global será sobre los 827.
#
# Para correr todas las publicaciones en server:
# ----------------------------------------------
# En docker-compose:
#
# SHAP_MAX_EXPLAIN_ROWS=827
# SHAP_BACKGROUND_SIZE=50
# SHAP_NSAMPLES=50
#
# o más pesado:
#
# SHAP_MAX_EXPLAIN_ROWS=827
# SHAP_BACKGROUND_SIZE=100
# SHAP_NSAMPLES=100
# =============================================================================

@app.post("/explain")
def explain(request: ExplainRequest):
    """
    Explica productos usando SHAP.

    Actualmente es síncrono:
        la API espera hasta que SHAP termina.

    Para pocos productos está bien.
    Para 827 productos puede demorar mucho.

    Más adelante conviene hacer:
        /explain-job
        /explain-status/{job_id}
        /explain-result/{job_id}
    """

    # -------------------------------------------------------------------------
    # 1. Cargar artefactos.
    #
    # Si falta alguno, significa que no se ejecutó /train.
    # -------------------------------------------------------------------------
    try:
        model, background, meta = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # -------------------------------------------------------------------------
    # 2. Convertir productos recibidos a DataFrame.
    # -------------------------------------------------------------------------
    df_original = rows_to_dataframe(request.rows)
    total_rows_received = len(df_original)

    # -------------------------------------------------------------------------
    # 3. Limitar cantidad de productos explicados.
    #
    # LOCAL:
    #   SHAP_MAX_EXPLAIN_ROWS=10
    #
    # SERVER:
    #   SHAP_MAX_EXPLAIN_ROWS=827
    #
    # Si mandas 827 y SHAP_MAX_EXPLAIN_ROWS=10:
    #   productos_recibidos = 827
    #   productos_explicados = 10
    #   truncated = true
    #
    # Si mandas 827 y SHAP_MAX_EXPLAIN_ROWS=827:
    #   productos_recibidos = 827
    #   productos_explicados = 827
    #   truncated = false
    # -------------------------------------------------------------------------
    df = df_original.head(SHAP_MAX_EXPLAIN_ROWS).copy()
    X = df[FEATURE_COLUMNS].copy()

    # -------------------------------------------------------------------------
    # 4. Crear funciones de predicción compatibles con AutoSklearn y SHAP.
    # -------------------------------------------------------------------------
    predict_fn = build_predict_fn(model)
    predict_proba_fn = build_predict_proba_fn(model)

    # -------------------------------------------------------------------------
    # 5. Predecir categoría y probabilidades.
    #
    # predictions:
    #   ["A", "B", "C", ...]
    #
    # probabilities:
    #   [[0.7, 0.2, 0.1], ...]
    # -------------------------------------------------------------------------
    predictions = predict_fn(X)
    probabilities = predict_proba_fn(X)

    # -------------------------------------------------------------------------
    # 6. Obtener clases conocidas por el modelo.
    #
    # Normalmente:
    #   ["A", "B", "C"]
    # -------------------------------------------------------------------------
    classes = [str(c) for c in model.classes_]

    # -------------------------------------------------------------------------
    # 7. Medir tiempo SHAP.
    # -------------------------------------------------------------------------
    start = time.time()

    # -------------------------------------------------------------------------
    # 8. Crear KernelExplainer.
    #
    # KernelExplainer es general, sirve para modelos complejos como ensambles.
    #
    # Desventaja:
    #   es lento.
    #
    # Usa:
    #   predict_proba_fn como función a explicar.
    #   background como referencia.
    # -------------------------------------------------------------------------
    explainer = shap.KernelExplainer(
        predict_proba_fn,
        background,
    )

    # -------------------------------------------------------------------------
    # 9. Calcular valores SHAP.
    #
    # Este es el paso caro.
    #
    # Costo aproximado aumenta con:
    #   - número de productos X
    #   - tamaño del background
    #   - nsamples
    #
    # MODIFICAR EN SERVER:
    # --------------------
    # SHAP_NSAMPLES=50
    # SHAP_NSAMPLES=100
    #
    # Mientras mayor, más lento pero más estable.
    # -------------------------------------------------------------------------
    shap_raw = explainer.shap_values(
        X,
        nsamples=SHAP_NSAMPLES,
    )

    shap_seconds = round(time.time() - start, 3)

    # -------------------------------------------------------------------------
    # 10. Normalizar formato de salida de SHAP.
    #
    # SHAP puede devolver:
    #   - lista por clase
    #   - array 2D
    #   - array 3D
    #
    # normalize_shap_values deja la salida en formato estándar.
    # -------------------------------------------------------------------------
    shap_values = normalize_shap_values(shap_raw)

    # -------------------------------------------------------------------------
    # 11. Construir explicaciones locales.
    #
    # Una explicación local responde:
    #
    #   ¿Por qué este producto quedó como A/B/C?
    #
    # Para cada producto:
    #   - se toma la clase predicha
    #   - se toman los SHAP values de esa clase
    #   - se ordenan las variables por impacto
    # -------------------------------------------------------------------------
    predicciones = []

    for i in range(len(X)):
        prediction = str(predictions[i])

        # Buscar índice de la clase predicha.
        # Ejemplo:
        #   classes = ["A", "B", "C"]
        #   prediction = "B"
        #   class_index = 1
        class_index = classes.index(prediction) if prediction in classes else 0

        # Si hay una sola salida, se toma [:, :, 0].
        # Si es multiclase, se toma la clase predicha.
        if shap_values.shape[-1] == 1:
            local_values = shap_values[i, :, 0]
        else:
            local_values = shap_values[i, :, class_index]

        # Armar lista de contribuciones:
        #
        # [
        #   {
        #     "feature": "stock_actual",
        #     "feature_value": 42,
        #     "shap_value": 0.29,
        #     "direction": "sube"
        #   },
        #   ...
        # ]
        contribuciones = build_local_contributions(
            feature_columns=FEATURE_COLUMNS,
            feature_values=X.iloc[i].tolist(),
            shap_values=local_values.tolist(),
        )

        # Convertir probabilidades a diccionario legible:
        #
        # {
        #   "A": 0.72,
        #   "B": 0.14,
        #   "C": 0.13
        # }
        proba_row = probabilities[i]
        probabilidades = {
            classes[j]: float(proba_row[j])
            for j in range(len(classes))
        }

        # Agregar resultado del producto.
        predicciones.append(
            {
                "publication_id": df.iloc[i]["publication_id"],
                "prediccion": prediction,
                "probabilidades": probabilidades,
                "top_contribuciones": contribuciones[: request.top_n],
                "contribuciones": contribuciones,
            }
        )

    # -------------------------------------------------------------------------
    # 12. Construir importancia global.
    #
    # La importancia global se calcula usando mean(abs(SHAP)).
    #
    # Si explicaste 10 productos:
    #   importancia global de esos 10.
    #
    # Si explicaste 827:
    #   importancia global de los 827.
    # -------------------------------------------------------------------------
    importancia_global = build_global_importance(
        feature_columns=FEATURE_COLUMNS,
        shap_values=shap_values,
    )

    # -------------------------------------------------------------------------
    # 13. Respuesta final.
    #
    # shap_config es clave para saber:
    #   - cuántos productos llegaron
    #   - cuántos se explicaron
    #   - si se recortó la entrada
    #   - cuánto demoró SHAP
    # -------------------------------------------------------------------------
    return {
        "mensaje": "Explicaciones generadas correctamente",
        "modelo": "AutoSklearnClassifier",
        "metrics_entrenamiento": meta.get("metrics"),
        "model_info": meta.get("model_info"),
        "shap_config": {
            "explainer": "KernelExplainer",
            "background_rows": int(len(background)),
            "nsamples": SHAP_NSAMPLES,
            "productos_recibidos": int(total_rows_received),
            "productos_explicados": int(len(X)),
            "limite_productos_explicados": SHAP_MAX_EXPLAIN_ROWS,
            "shap_seconds": shap_seconds,
            "truncated": total_rows_received > len(X),
        },
        "importancia_global": importancia_global,
        "predicciones": predicciones,
    }