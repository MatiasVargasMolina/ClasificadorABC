from __future__ import annotations

import hashlib
import json
import platform
import random
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import requests


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "input_request.json"
OUTPUT_ROOT = ROOT / "artifacts" / "service_performance"
API_URL = "http://127.0.0.1:8000"

CLASSIFICATION_SIZES = [20, 50, 100, 250, 500, 832]
SHAP_SIZES = [1, 10, 50, 100, 832]

CLASSIFICATION_REPETITIONS = 10
SHAP_REPETITIONS = 3
SHAP_FULL_REPETITIONS = 1

SEED = 42
TIMEOUT = 7200


def run_id() -> str:
    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    return (
        f"services-performance-"
        f"{stamp}-"
        f"{uuid.uuid4().hex[:10]}"
    )


def save_json(
    path: Path,
    value: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_products() -> list[
    dict[str, Any]
]:
    data = json.loads(
        INPUT.read_text(
            encoding="utf-8-sig"
        )
    )

    products = data.get(
        "productos"
    )

    if (
        not isinstance(products, list)
        or len(products) != 832
    ):
        raise ValueError(
            "data/input_request.json "
            "debe contener exactamente "
            "832 publicaciones."
        )

    ids = [
        str(
            product.get(
                "publication_id",
                "",
            )
        )
        for product in products
    ]

    if (
        not all(ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(
            "Todos los publication_id "
            "deben existir y ser únicos."
        )

    return products


def samples(
    products: list[dict[str, Any]],
    sizes: list[int],
) -> dict[
    int,
    list[dict[str, Any]],
]:
    indexes = list(
        range(len(products))
    )

    random.Random(
        SEED
    ).shuffle(indexes)

    result = {
        size: [
            products[index]
            for index in indexes[:size]
        ]
        for size in sizes
    }

    # Para el lote completo se conserva
    # el orden oficial del dataset.
    if len(products) in result:
        result[len(products)] = list(
            products
        )

    return result


def command(
    command_args: list[str],
) -> str | None:
    try:
        result = subprocess.run(
            command_args,
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )

        return result.stdout.strip()

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            registry_path = (
                r"HARDWARE\DESCRIPTION"
                r"\System\CentralProcessor\0"
            )

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                registry_path,
            ) as key:
                value = winreg.QueryValueEx(
                    key,
                    "ProcessorNameString",
                )[0]

                return str(
                    value
                ).strip()

        except OSError:
            pass

    return (
        platform.processor()
        or "No informado"
    )


def versions(
    names: list[str],
) -> dict[str, str | None]:
    result: dict[
        str,
        str | None,
    ] = {}

    for name in names:
        try:
            result[name] = (
                metadata.version(name)
            )

        except (
            metadata.PackageNotFoundError
        ):
            result[name] = None

    return result


def environment(
    execution_id: str,
) -> dict[str, Any]:
    git_status = command(
        [
            "git",
            "status",
            "--short",
        ]
    )

    worker_requirements = (
        ROOT
        / "autosklearn_worker"
        / "requirements.txt"
    )

    return {
        "id_ejecucion": (
            execution_id
        ),
        "fecha_inicio_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "dataset_sha256": (
            hashlib.sha256(
                INPUT.read_bytes()
            ).hexdigest()
        ),
        "hardware": {
            "cpu": cpu_name(),
            "nucleos_fisicos": (
                psutil.cpu_count(
                    logical=False
                )
            ),
            "nucleos_logicos": (
                psutil.cpu_count(
                    logical=True
                )
            ),
            "ram_gb": round(
                psutil.virtual_memory().total
                / 1024**3,
                2,
            ),
            "arquitectura": (
                platform.machine()
            ),
        },
        "sistema_operativo": (
            platform.platform()
        ),
        "python": (
            platform.python_version()
        ),
        "versiones_host": versions(
            [
                "fastapi",
                "uvicorn",
                "pydantic",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "optuna",
                "requests",
                "psutil",
                "pytest",
            ]
        ),
        "versiones_worker": (
            worker_requirements.read_text(
                encoding="utf-8"
            )
        ),
        "docker": command(
            [
                "docker",
                "--version",
            ]
        ),
        "docker_compose": command(
            [
                "docker",
                "compose",
                "version",
            ]
        ),
        "commit": command(
            [
                "git",
                "rev-parse",
                "HEAD",
            ]
        ),
        "rama": command(
            [
                "git",
                "branch",
                "--show-current",
            ]
        ),
        "git_limpio": (
            git_status == ""
            if git_status is not None
            else None
        ),
        "cambios_pendientes": (
            git_status.splitlines()
            if git_status
            else []
        ),
        "protocolo": {
            "clasificacion_tamanos": (
                CLASSIFICATION_SIZES
            ),
            "clasificacion_repeticiones": (
                CLASSIFICATION_REPETITIONS
            ),
            "clasificacion_calentamientos_por_tamano": 1,
            "optuna_repeticiones": 1,
            "entrenamiento_repeticiones": 1,
            "entrenamiento_time_left_s": 600,
            "entrenamiento_per_run_limit_s": 60,
            "shap_tamanos": (
                SHAP_SIZES
            ),
            "shap_repeticiones": (
                SHAP_REPETITIONS
            ),
            "shap_repeticiones_832": (
                SHAP_FULL_REPETITIONS
            ),
            "shap_calentamientos": 1,
            "semilla_muestreo": (
                SEED
            ),
            "memoria": (
                "RSS/Working Set de la "
                "API principal; no incluye "
                "el worker Docker."
            ),
        },
    }


def port_open(
    port: int,
) -> bool:
    try:
        with socket.create_connection(
            (
                "127.0.0.1",
                port,
            ),
            timeout=0.3,
        ):
            return True

    except OSError:
        return False


def verify_runtime() -> None:
    if not port_open(8000):
        raise RuntimeError(
            "Falta la API principal en "
            "127.0.0.1:8000."
        )

    if not port_open(8010):
        raise RuntimeError(
            "Falta el worker Docker en "
            "127.0.0.1:8010."
        )


def start_worker() -> None:
    compose_file = (
        ROOT
        / "docker-compose.autosklearn.yml"
    )

    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "up",
            "-d",
            "--build",
            "autosklearn-worker",
        ],
        cwd=ROOT,
        check=False,
    )

    if result.returncode:
        raise RuntimeError(
            "No fue posible iniciar "
            "el worker Docker."
        )

    deadline = (
        time.monotonic()
        + 300
    )

    while not port_open(8010):
        if (
            time.monotonic()
            >= deadline
        ):
            raise RuntimeError(
                "El worker Docker no abrió "
                "el puerto 8010."
            )

        time.sleep(0.5)


def api_pid() -> int:
    pids = [
        connection.pid
        for connection
        in psutil.net_connections(
            kind="inet"
        )
        if (
            connection.pid
            and connection.laddr
            and connection.laddr.port
            == 8000
            and connection.status
            == psutil.CONN_LISTEN
        )
    ]

    if not pids:
        raise RuntimeError(
            "No fue posible encontrar "
            "el PID de Uvicorn en "
            "el puerto 8000."
        )

    return max(
        set(pids),
        key=lambda pid: (
            psutil.Process(
                pid
            ).memory_info().rss
        ),
    )


def tree_memory_mb(
    pid: int,
) -> float:
    root_process = psutil.Process(
        pid
    )

    processes = [
        root_process,
        *root_process.children(
            recursive=True
        ),
    ]

    total = 0

    for process in processes:
        try:
            total += (
                process.memory_info().rss
            )

        except psutil.Error:
            pass

    return total / 1024**2


class MemoryMonitor:
    def __init__(
        self,
        pid: int,
    ) -> None:
        self.pid = pid
        self.values: list[
            float
        ] = []

        self.stop_event = (
            threading.Event()
        )

    def _sample(self) -> None:
        while not self.stop_event.wait(
            0.01
        ):
            try:
                self.values.append(
                    tree_memory_mb(
                        self.pid
                    )
                )

            except psutil.Error:
                pass

    def start(self) -> float:
        baseline = tree_memory_mb(
            self.pid
        )

        self.values = [
            baseline
        ]

        self.stop_event.clear()

        self.thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )

        self.thread.start()

        return baseline

    def stop(self) -> float:
        self.stop_event.set()

        self.thread.join(
            timeout=2
        )

        self.values.append(
            tree_memory_mb(
                self.pid
            )
        )

        return max(
            self.values
        )


def request(
    session: requests.Session,
    endpoint: str,
    payload: dict[str, Any],
    params: (
        dict[str, Any]
        | None
    ) = None,
    memory_pid: (
        int
        | None
    ) = None,
) -> tuple[
    dict[str, Any],
    float,
    int,
    float | None,
    float | None,
]:
    monitor = (
        MemoryMonitor(
            memory_pid
        )
        if memory_pid
        else None
    )

    baseline = (
        monitor.start()
        if monitor
        else None
    )

    started = (
        time.perf_counter()
    )

    try:
        response = session.post(
            API_URL + endpoint,
            json=payload,
            params=params,
            timeout=TIMEOUT,
        )

        elapsed = (
            time.perf_counter()
            - started
        )

    finally:
        peak = (
            monitor.stop()
            if monitor
            else None
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"{endpoint} respondió "
            f"HTTP {response.status_code}: "
            f"{response.text[:1500]}"
        )

    return (
        response.json(),
        elapsed,
        response.status_code,
        baseline,
        peak,
    )


def base_record(
    execution_id: str,
    service: str,
    endpoint: str,
    size: int,
    repetition: int,
    status: int,
    elapsed: float,
    internal: (
        float
        | None
    ) = None,
    baseline: (
        float
        | None
    ) = None,
    peak: (
        float
        | None
    ) = None,
) -> dict[str, Any]:
    return {
        "id_ejecucion": (
            execution_id
        ),
        "servicio": service,
        "endpoint": endpoint,
        "publicaciones": size,
        "repeticion": (
            repetition
        ),
        "http_status": status,
        "latencia_total_s": round(
            elapsed,
            9,
        ),
        "tiempo_interno_s": (
            round(
                float(internal),
                9,
            )
            if internal is not None
            else None
        ),
        "publicaciones_por_segundo": (
            round(
                size / elapsed,
                6,
            )
        ),
        "memoria_base_mb": (
            round(
                baseline,
                3,
            )
            if baseline is not None
            else None
        ),
        "memoria_pico_mb": (
            round(
                peak,
                3,
            )
            if peak is not None
            else None
        ),
        "incremento_memoria_mb": (
            round(
                max(
                    0.0,
                    peak - baseline,
                ),
                3,
            )
            if (
                peak is not None
                and baseline is not None
            )
            else None
        ),
    }


def save_service(
    run_dir: Path,
    name: str,
    records: list[
        dict[str, Any]
    ],
) -> pd.DataFrame:
    detail = pd.DataFrame(
        records
    )

    for column in (
        "tiempo_interno_s",
        "memoria_pico_mb",
        "publicaciones_por_segundo",
    ):
        detail[column] = (
            pd.to_numeric(
                detail[column],
                errors="coerce",
            )
        )

    detail.to_csv(
        run_dir
        / f"{name}_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        detail
        .groupby(
            "publicaciones",
            as_index=False,
        )
        .agg(
            repeticiones=(
                "repeticion",
                "count",
            ),
            latencia_mediana_s=(
                "latencia_total_s",
                "median",
            ),
            latencia_min_s=(
                "latencia_total_s",
                "min",
            ),
            latencia_max_s=(
                "latencia_total_s",
                "max",
            ),
            tiempo_interno_mediana_s=(
                "tiempo_interno_s",
                "median",
            ),
            memoria_maxima_mb=(
                "memoria_pico_mb",
                "max",
            ),
            publicaciones_por_segundo_mediana=(
                "publicaciones_por_segundo",
                "median",
            ),
        )
    )

    summary.insert(
        0,
        "servicio",
        records[0]["servicio"],
    )

    summary.insert(
        1,
        "endpoint",
        records[0]["endpoint"],
    )

    summary.to_csv(
        run_dir
        / f"{name}_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return summary


def classification_test(
    session: requests.Session,
    execution_id: str,
    products: list[
        dict[str, Any]
    ],
    pid: int,
    run_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    records: list[
        dict[str, Any]
    ] = []

    full: dict[
        str,
        Any,
    ] = {}

    batches = samples(
        products,
        CLASSIFICATION_SIZES,
    )

    for size, batch in batches.items():
        print(
            "\nClasificación:",
            size,
            "publicaciones",
        )

        warmup, *_ = request(
            session,
            "/api/clasificar",
            {
                "productos": batch,
            },
        )

        if (
            len(
                warmup.get(
                    "resultados",
                    [],
                )
            )
            != size
        ):
            raise RuntimeError(
                "Falló el calentamiento "
                "de clasificación."
            )

        for repetition in range(
            1,
            CLASSIFICATION_REPETITIONS
            + 1,
        ):
            (
                data,
                elapsed,
                status,
                baseline,
                peak,
            ) = request(
                session,
                "/api/clasificar",
                {
                    "productos": batch,
                },
                memory_pid=pid,
            )

            results = data.get(
                "resultados",
                [],
            )

            if (
                len(results) != size
                or data.get(
                    "productos_invalidos"
                )
            ):
                raise RuntimeError(
                    "Clasificación incompleta "
                    f"para n={size}."
                )

            counts = Counter(
                row["categoria"]
                for row in results
            )

            diagnostic = data.get(
                "diagnostico",
                {},
            )

            record = base_record(
                execution_id,
                "clasificacion",
                "/api/clasificar",
                size,
                repetition,
                status,
                elapsed,
                baseline=baseline,
                peak=peak,
            )

            record.update(
                {
                    "conteo_A": (
                        counts.get(
                            "A",
                            0,
                        )
                    ),
                    "conteo_B": (
                        counts.get(
                            "B",
                            0,
                        )
                    ),
                    "conteo_C": (
                        counts.get(
                            "C",
                            0,
                        )
                    ),
                    "iteraciones": (
                        diagnostic.get(
                            "iteraciones"
                        )
                    ),
                    "inercia": (
                        diagnostic.get(
                            "inertia"
                        )
                    ),
                    "silueta": (
                        diagnostic
                        .get(
                            "metricas",
                            {},
                        )
                        .get(
                            "silhouette"
                        )
                    ),
                }
            )

            records.append(
                record
            )

            print(
                f"  {repetition:02d}/"
                f"{CLASSIFICATION_REPETITIONS}: "
                f"{elapsed:.6f} s | "
                f"{peak:.2f} MB"
            )

            if size == 832:
                full = data

    save_json(
        run_dir
        / "classification_response_832.json",
        full,
    )

    return (
        save_service(
            run_dir,
            "classification",
            records,
        ),
        full,
    )


def optimization_test(
    session: requests.Session,
    execution_id: str,
    products: list[
        dict[str, Any]
    ],
    pid: int,
    run_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    print(
        "\nOptimización Optuna"
    )

    (
        data,
        elapsed,
        status,
        baseline,
        peak,
    ) = request(
        session,
        "/optimization/optuna",
        {
            "productos": products,
        },
        memory_pid=pid,
    )

    if data.get(
        "best_params"
    ) is None:
        raise RuntimeError(
            "Optuna no devolvió "
            "best_params."
        )

    record = base_record(
        execution_id,
        "optimizacion_optuna",
        "/optimization/optuna",
        832,
        1,
        status,
        elapsed,
        baseline=baseline,
        peak=peak,
    )

    record[
        "publicaciones_por_segundo"
    ] = None

    record.update(
        {
            "best_value": (
                data.get(
                    "best_value"
                )
            ),
            "best_trial": (
                data.get(
                    "best_trial"
                )
            ),
            "best_params": (
                json.dumps(
                    data.get(
                        "best_params"
                    ),
                    ensure_ascii=False,
                )
            ),
            "trials": len(
                data.get(
                    "trials",
                    [],
                )
            ),
        }
    )

    print(
        f"  {elapsed:.3f} s | "
        f"{peak:.2f} MB"
    )

    save_json(
        run_dir
        / "optimization_response.json",
        data,
    )

    return (
        save_service(
            run_dir,
            "optimization",
            [record],
        ),
        data,
    )


def training_test(
    session: requests.Session,
    execution_id: str,
    products: list[
        dict[str, Any]
    ],
    run_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    print(
        "\nEntrenamiento AutoSklearn"
    )

    (
        data,
        elapsed,
        status,
        _,
        _,
    ) = request(
        session,
        (
            "/api/explainability/"
            "autosklearn/train"
        ),
        {
            "productos": products,
        },
        params={
            "time_left_for_this_task": 600,
            "per_run_time_limit": 60,
        },
    )

    worker = data.get(
        "autosklearn",
        {},
    )

    metrics = worker.get(
        "metrics",
        {},
    )

    if not worker.get(
        "id_ejecucion"
    ):
        raise RuntimeError(
            "Entrenamiento no devolvió "
            "id_ejecucion."
        )

    internal = metrics.get(
        "training_seconds"
    )

    record = base_record(
        execution_id,
        "entrenamiento_autosklearn",
        (
            "/api/explainability/"
            "autosklearn/train"
        ),
        832,
        1,
        status,
        elapsed,
        internal=internal,
    )

    record[
        "publicaciones_por_segundo"
    ] = None

    record.update(
        {
            "id_entrenamiento": (
                worker.get(
                    "id_ejecucion"
                )
            ),
            "version_artefacto": (
                worker.get(
                    "version_artefacto"
                )
            ),
            "accuracy": (
                metrics.get(
                    "accuracy"
                )
            ),
            "balanced_accuracy": (
                metrics.get(
                    "balanced_accuracy"
                )
            ),
            "macro_f1": (
                metrics.get(
                    "macro_f1"
                )
            ),
        }
    )

    print(
        f"  total={elapsed:.3f} s | "
        "entrenamiento="
        f"{float(internal):.3f} s"
    )

    save_json(
        run_dir
        / "training_response.json",
        data,
    )

    return (
        save_service(
            run_dir,
            "training",
            [record],
        ),
        data,
    )


def shap_test(
    session: requests.Session,
    execution_id: str,
    products: list[
        dict[str, Any]
    ],
    run_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    print(
        "\nExplicabilidad SHAP"
    )

    batches = samples(
        products,
        SHAP_SIZES,
    )

    warmup, *_ = request(
        session,
        (
            "/api/explainability/"
            "autosklearn/explain"
        ),
        {
            "productos": (
                batches[1]
            ),
        },
        params={
            "top_n": 3,
        },
    )

    if (
        len(
            warmup.get(
                "predicciones",
                [],
            )
        )
        != 1
    ):
        raise RuntimeError(
            "Falló el calentamiento SHAP."
        )

    records: list[
        dict[str, Any]
    ] = []

    full: dict[
        str,
        Any,
    ] = {}

    for size, batch in batches.items():
        repetitions = (
            SHAP_FULL_REPETITIONS
            if size == 832
            else SHAP_REPETITIONS
        )

        for repetition in range(
            1,
            repetitions + 1,
        ):
            (
                data,
                elapsed,
                status,
                _,
                _,
            ) = request(
                session,
                (
                    "/api/explainability/"
                    "autosklearn/explain"
                ),
                {
                    "productos": batch,
                },
                params={
                    "top_n": 3,
                },
            )

            config = data.get(
                "shap_config",
                {},
            )

            predictions = data.get(
                "predicciones",
                [],
            )

            if (
                len(predictions) != size
                or config.get(
                    "truncated"
                )
            ):
                raise RuntimeError(
                    "SHAP no explicó "
                    "completamente "
                    f"n={size}."
                )

            internal = config.get(
                "shap_seconds"
            )

            agreement = data.get(
                "resumen_concordancia",
                {},
            )

            additivity = data.get(
                "validacion_aditividad",
                {},
            )

            record = base_record(
                execution_id,
                "explicabilidad_shap",
                (
                    "/api/explainability/"
                    "autosklearn/explain"
                ),
                size,
                repetition,
                status,
                elapsed,
                internal=internal,
            )

            record.update(
                {
                    "id_shap": (
                        data.get(
                            "id_ejecucion"
                        )
                    ),
                    "id_entrenamiento": (
                        data.get(
                            "id_ejecucion_entrenamiento"
                        )
                    ),
                    "version_artefacto": (
                        data.get(
                            "version_artefacto"
                        )
                    ),
                    "coincidencias": (
                        agreement.get(
                            "coincidencias"
                        )
                    ),
                    "discrepancias": (
                        agreement.get(
                            "discrepancias"
                        )
                    ),
                    "porcentaje_concordancia": (
                        agreement.get(
                            "porcentaje_concordancia"
                        )
                    ),
                    "aditividad_cumplida": (
                        additivity.get(
                            "cumple_tolerancia"
                        )
                    ),
                    "error_aditividad_maximo": (
                        additivity.get(
                            "max_absolute_error"
                        )
                    ),
                }
            )

            records.append(
                record
            )

            print(
                f"  n={size}, "
                f"{repetition}/"
                f"{repetitions}: "
                f"total={elapsed:.3f} s | "
                f"SHAP={float(internal):.3f} s"
            )

            if size == 832:
                full = data

            time.sleep(5)

    save_json(
        run_dir
        / "shap_response_832.json",
        full,
    )

    return (
        save_service(
            run_dir,
            "shap",
            records,
        ),
        full,
    )


def traceability(
    classification: dict[
        str,
        Any,
    ],
    shap: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    class_map = {
        row["publication_id"]: (
            row["categoria"]
        )
        for row
        in classification.get(
            "resultados",
            [],
        )
    }

    shap_map = {
        row["publication_id"]: (
            row.get(
                "categoria_ss_ekmeans"
            )
        )
        for row
        in shap.get(
            "predicciones",
            [],
        )
    }

    mismatches = [
        publication_id
        for publication_id
        in class_map
        if (
            class_map[
                publication_id
            ]
            != shap_map.get(
                publication_id
            )
        )
    ]

    result = {
        "publicaciones": (
            len(class_map)
        ),
        "discrepancias_categoria_ss_ekmeans": (
            len(mismatches)
        ),
        "ids_discrepantes": (
            mismatches
        ),
        "cumple": (
            len(class_map)
            == len(shap_map)
            == 832
            and not mismatches
        ),
    }

    if not result["cumple"]:
        raise RuntimeError(
            "La clasificación y las "
            "categorías SS-EKMeans de "
            "SHAP no coinciden."
        )

    return result


def main() -> None:
    products = load_products()

    execution_id = run_id()

    run_dir = (
        OUTPUT_ROOT
        / execution_id
    )

    run_dir.mkdir(
        parents=True
    )

    env = environment(
        execution_id
    )

    save_json(
        run_dir
        / "environment.json",
        env,
    )

    print(
        "ID:",
        execution_id,
    )

    print(
        "Resultados:",
        run_dir,
    )

    print(
        "No se consulta ni se mide "
        "el endpoint de disponibilidad."
    )

    start_worker()

    verify_runtime()

    pid = api_pid()

    print(
        "PID de Uvicorn utilizado "
        "para memoria:",
        pid,
    )

    env["api_principal"] = {
        "pid": pid,
        "comando": (
            psutil.Process(
                pid
            ).cmdline()
        ),
    }

    save_json(
        run_dir
        / "environment.json",
        env,
    )

    session = requests.Session()

    try:
        (
            class_summary,
            class_response,
        ) = classification_test(
            session,
            execution_id,
            products,
            pid,
            run_dir,
        )

        (
            optuna_summary,
            optuna_response,
        ) = optimization_test(
            session,
            execution_id,
            products,
            pid,
            run_dir,
        )

        (
            training_summary,
            training_response,
        ) = training_test(
            session,
            execution_id,
            products,
            run_dir,
        )

        (
            shap_summary,
            shap_response,
        ) = shap_test(
            session,
            execution_id,
            products,
            run_dir,
        )

    finally:
        session.close()

    trace = traceability(
        class_response,
        shap_response,
    )

    save_json(
        run_dir
        / "traceability.json",
        trace,
    )

    all_summary = pd.concat(
        [
            class_summary,
            optuna_summary,
            training_summary,
            shap_summary,
        ],
        ignore_index=True,
    )

    all_summary.to_csv(
        run_dir
        / "all_services_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    services_for_json = (
        all_summary
        .astype(object)
        .where(
            pd.notna(
                all_summary
            ),
            None,
        )
        .to_dict(
            "records"
        )
    )

    save_json(
        run_dir
        / "all_services_summary.json",
        {
            "id_ejecucion": (
                execution_id
            ),
            "servicios": (
                services_for_json
            ),
            "trazabilidad": (
                trace
            ),
        },
    )

    # Respuestas estables utilizadas
    # por los scripts de figuras.
    save_json(
        ROOT
        / "data"
        / "classification_response.json",
        class_response,
    )

    save_json(
        ROOT
        / "data"
        / "optuna_result.json",
        optuna_response,
    )

    save_json(
        ROOT
        / "data"
        / "autosklearn_train_response.json",
        training_response,
    )

    save_json(
        ROOT
        / "data"
        / "shap_explain_response.json",
        shap_response,
    )

    env["fecha_fin_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_json(
        run_dir
        / "environment.json",
        env,
    )

    print(
        "\nPruebas completadas. "
        "Archivos:"
    )

    for path in sorted(
        run_dir.iterdir()
    ):
        print(
            "  -",
            path,
        )


if __name__ == "__main__":
    main()