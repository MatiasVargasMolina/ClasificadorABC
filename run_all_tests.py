from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIRECTORY = Path(
    __file__
).resolve().parent

TEST_DIRECTORY = (
    ROOT_DIRECTORY / "test"
)

WORKER_TEST_DIRECTORY = (
    TEST_DIRECTORY / "worker"
)

COMPOSE_FILE = (
    ROOT_DIRECTORY
    / "docker-compose.autosklearn.yml"
)

WORKER_SERVICE = "autosklearn-worker"


def run_command(
    title: str,
    command: Sequence[str],
) -> bool:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print(" ".join(command))
    print()

    completed = subprocess.run(
        list(command),
        cwd=ROOT_DIRECTORY,
        check=False,
    )

    if completed.returncode != 0:
        print()
        print(
            f"ERROR: {title} terminó con código "
            f"{completed.returncode}."
        )
        return False

    print()
    print(f"OK: {title}")
    return True


def validate_environment() -> bool:
    if not TEST_DIRECTORY.exists():
        print(
            f"ERROR: No existe la carpeta "
            f"{TEST_DIRECTORY}."
        )
        return False

    if not WORKER_TEST_DIRECTORY.exists():
        print(
            f"ERROR: No existe la carpeta "
            f"{WORKER_TEST_DIRECTORY}."
        )
        return False

    if not COMPOSE_FILE.exists():
        print(
            f"ERROR: No existe "
            f"{COMPOSE_FILE}."
        )
        return False

    if shutil.which("docker") is None:
        print(
            "ERROR: Docker no está disponible "
            "en el PATH."
        )
        return False

    return True


def run_main_application_tests() -> bool:
    return run_command(
        title=(
            "Pruebas de la aplicación principal"
        ),
        command=[
            sys.executable,
            "-m",
            "pytest",
            str(TEST_DIRECTORY),
            "-q",
            "--ignore",
            str(WORKER_TEST_DIRECTORY),
        ],
    )


def build_worker_image() -> bool:
    return run_command(
        title=(
            "Construcción del worker de "
            "explicabilidad"
        ),
        command=[
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "build",
            WORKER_SERVICE,
        ],
    )


def run_worker_tests() -> bool:
    repository_mount = (
        f"{ROOT_DIRECTORY}:/repo"
    )

    return run_command(
        title=(
            "Pruebas del flujo real de "
            "Kernel SHAP"
        ),
        command=[
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "run",
            "--rm",
            "--no-deps",
            "-v",
            repository_mount,
            "--entrypoint",
            "sh",
            WORKER_SERVICE,
            "-c",
            (
                "cd /repo && "
                "PYTHONPATH=/repo "
                "python -m pytest "
                "-q test/worker"
            ),
        ],
    )


def main() -> int:
    if not validate_environment():
        return 1

    if not run_main_application_tests():
        return 1

    if not build_worker_image():
        return 1

    if not run_worker_tests():
        return 1

    print()
    print("=" * 70)
    print(
        "TODAS LAS PRUEBAS TERMINARON "
        "CORRECTAMENTE"
    )
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())