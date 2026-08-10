"""Persistencia de corridas: identidad por contenido, procedencia y caché.

`exists()` solo devuelve verdadero cuando existe `metrics.json`, que se
escribe al final. Una corrida interrumpida deja el directorio a medias y
no se toma como cacheada.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.job import Artifacts, Job
from core.model import ModelSpec

_CHUNK = 1 << 20


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def compute_run_id(job: Job, spec: ModelSpec) -> str:
    """Identidad por contenido. No depende del tiempo ni de rutas."""
    payload = {
        "model": spec.name,
        "revision": spec.revision,
        "params": job.params,
        "export": job.export,
        "seed": job.seed,
        "inputs": {key: _hash_file(path) for key, path in sorted(job.inputs.items())},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:16]


def _repo_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_provenance(job: Job, spec: ModelSpec, backend_name: str) -> dict[str, Any]:
    return {
        "model": spec.name,
        "model_revision": spec.revision,
        "docker_image": spec.docker_image,
        "backend": backend_name,
        "seed": job.seed,
        "repo_sha": _repo_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # Rastrear qué run_id ya fueron iniciados en este proceso.
        # Permite que write_job() se llame múltiples veces en el mismo intento
        # sin destruir outputs, pero sigue limpiando basura de intentos anteriores (crashes).
        self._initiated_run_ids: set[str] = set()

    def _dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        """Crear directorios para una corrida. Solo mkdir -p, idempotente, nunca destructivo."""
        base = self._dir(run_id)
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        (base / "outputs").mkdir(parents=True, exist_ok=True)
        return base

    def exists(self, run_id: str) -> bool:
        """Una corrida existe si:
        - metrics.json es JSON válido (indica completitud)
        - AND .in-progress no existe (fue eliminado en write_metrics)

        Así se distinguen estados:
        - Corrida completada: metrics.json válido, sin .in-progress
        - Corrida en progreso: .in-progress presente
        - Corrida fallida (crash): .in-progress presente, metrics.json corrupto o ausente
        - Corrida inexistente: ni metrics ni .in-progress
        """
        base_dir = self._dir(run_id)
        metrics_path = base_dir / "metrics.json"
        in_progress_marker = base_dir / ".in-progress"

        # Debe existir metrics.json válido
        if not metrics_path.is_file():
            return False
        try:
            json.loads(metrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            return False

        # AND no debe existir el marcador de "en progreso"
        return not in_progress_marker.is_file()

    def inputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "inputs"

    def outputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "outputs"

    def write_job(self, run_id: str, job: Job) -> None:
        """Escribir job.json. Se puede llamar múltiples veces en el mismo intento.

        La primera llamada para un run_id en este proceso:
        - Limpia restos de intentos anteriores fallidos (outputs viejos)
        - Marca que este intento está en progreso (.in-progress)

        Llamadas posteriores (dentro del mismo intento):
        - No limpian nada (es el mismo intento, no basura vieja)
        - Solo actualizan job.json
        """
        base = self.create(run_id)
        in_progress_marker = base / ".in-progress"

        # Si es la primera vez que escribimos este run_id en este proceso
        # (el run_id no está en _initiated_run_ids), es un reintento sobre basura vieja.
        # Limpiar y eliminar marcador antiguo.
        if run_id not in self._initiated_run_ids:
            if in_progress_marker.is_file():
                outputs = base / "outputs"
                if outputs.exists():
                    for file in outputs.iterdir():
                        if file.is_file():
                            file.unlink()
                in_progress_marker.unlink()
            # Marcar este run_id como iniciado en este proceso
            self._initiated_run_ids.add(run_id)

        # Escribir job.json y crear/recrear marcador de "en progreso"
        (base / "job.json").write_text(job.model_dump_json(indent=2))
        in_progress_marker.touch()

    def write_provenance(self, run_id: str, data: dict[str, Any]) -> None:
        (self.create(run_id) / "provenance.json").write_text(json.dumps(data, indent=2))

    def write_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        base = self.create(run_id)
        metrics_path = base / "metrics.json"
        temp_path = base / "metrics.json.tmp"
        # Escribir a temporal primero, luego reemplazar atómicamente
        temp_path.write_text(json.dumps(metrics, indent=2))
        os.replace(str(temp_path), str(metrics_path))
        # Eliminar marcador de "en progreso" ya que la corrida completó exitosamente
        (base / ".in-progress").unlink(missing_ok=True)

    def load_artifacts(self, run_id: str) -> Artifacts:
        """Cargar artefactos de una corrida.

        Lanza FileNotFoundError si la corrida no fue nunca iniciada (no existe job.json).
        Devuelve Artifacts con metrics={} y files={} vacío si la corrida no completó.
        """
        base_dir = self._dir(run_id)
        job_path = base_dir / "job.json"
        metrics_path = base_dir / "metrics.json"

        # Si no existe job.json, la corrida nunca fue iniciada: error
        if not job_path.is_file():
            raise FileNotFoundError(f"Run {run_id} not found")

        # Leer metrics si existe y es válido
        metrics = {}
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text())
            except (json.JSONDecodeError, OSError):
                # Si metrics está corrupto, devolver vacío (corrida falló)
                pass

        # Leer outputs si existen (no crear directorios)
        outputs = base_dir / "outputs"
        files = {}
        if outputs.is_dir():
            files = {path.name: path for path in sorted(outputs.iterdir()) if path.is_file()}

        return Artifacts(files=files, metrics=metrics)
