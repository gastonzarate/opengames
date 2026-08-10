"""Persistencia de corridas: identidad por contenido, procedencia y caché.

`exists()` solo devuelve verdadero cuando existe `metrics.json`, que se
escribe al final. Una corrida interrumpida deja el directorio a medias y
no se toma como cacheada.
"""

import hashlib
import json
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
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        (base / "outputs").mkdir(parents=True, exist_ok=True)
        return base

    def exists(self, run_id: str) -> bool:
        return (self._dir(run_id) / "metrics.json").is_file()

    def inputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "inputs"

    def outputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "outputs"

    def write_job(self, run_id: str, job: Job) -> None:
        (self.create(run_id) / "job.json").write_text(job.model_dump_json(indent=2))

    def write_provenance(self, run_id: str, data: dict[str, Any]) -> None:
        (self.create(run_id) / "provenance.json").write_text(json.dumps(data, indent=2))

    def write_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        (self.create(run_id) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def load_artifacts(self, run_id: str) -> Artifacts:
        metrics_path = self._dir(run_id) / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        outputs = self.outputs_dir(run_id)
        files = {path.name: path for path in sorted(outputs.iterdir()) if path.is_file()}
        return Artifacts(files=files, metrics=metrics)
