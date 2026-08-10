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

    def _dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        base = self._dir(run_id)
        in_progress_marker = base / ".in-progress"

        # Limpiar outputs de corridas anteriores (tanto fallidas como exitosas) en reintentos:
        # Si existe outputs pero no existe .in-progress, es un reintento y hay archivos
        # viejos que deben descartarse antes de la nueva corrida.
        outputs = base / "outputs"
        if outputs.exists() and not in_progress_marker.is_file():
            for file in outputs.iterdir():
                if file.is_file():
                    file.unlink()

        # Crear directorio y marcador de "en progreso"
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        outputs.mkdir(parents=True, exist_ok=True)
        if not in_progress_marker.is_file():
            in_progress_marker.touch()

        return base

    def exists(self, run_id: str) -> bool:
        metrics_path = self._dir(run_id) / "metrics.json"
        if not metrics_path.is_file():
            return False
        try:
            json.loads(metrics_path.read_text())
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def inputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "inputs"

    def outputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "outputs"

    def write_job(self, run_id: str, job: Job) -> None:
        (self.create(run_id) / "job.json").write_text(job.model_dump_json(indent=2))

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
        base_dir = self._dir(run_id)
        metrics_path = base_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        # No llamar a outputs_dir() para evitar crear directorios como efecto secundario
        outputs = base_dir / "outputs"
        files = {}
        if outputs.is_dir():
            files = {path.name: path for path in sorted(outputs.iterdir()) if path.is_file()}
        return Artifacts(files=files, metrics=metrics)
