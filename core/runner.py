"""Orquesta una corrida: valida, cachea, ejecuta y limpia."""

import shutil
import time
from dataclasses import dataclass

from core.backend import Backend
from core.job import Artifacts, Job, RunStatus
from core.registry import get_model
from core.runstore import RunStore, collect_provenance, compute_run_id


class InsufficientVram(RuntimeError):
    """El backend elegido no llega a la VRAM que el modelo declara."""


class GenerationFailed(RuntimeError):
    """El backend reportó estado FAILED."""


@dataclass
class RunResult:
    run_id: str
    artifacts: Artifacts
    cached: bool


def execute(
    job: Job,
    backend: Backend,
    store: RunStore,
    poll_interval: float = 0.0,
) -> RunResult:
    spec = get_model(job.model).describe()
    caps = backend.capabilities()

    if spec.min_vram_gb > caps.vram_gb:
        raise InsufficientVram(
            f"El modelo '{spec.name}' necesita {spec.min_vram_gb} GB de VRAM y el "
            f"backend '{caps.name}' ofrece {caps.vram_gb} GB. "
            f"Elegí un backend con más memoria o un modelo más chico."
        )

    run_id = compute_run_id(job, spec)
    if store.exists(run_id):
        return RunResult(run_id=run_id, artifacts=store.load_artifacts(run_id), cached=True)

    store.create(run_id)
    store.write_job(run_id, job)
    store.write_provenance(run_id, collect_provenance(job, spec, caps.name))
    for name, source in job.inputs.items():
        shutil.copy2(source, store.inputs_dir(run_id) / source.name)

    handle = backend.submit(job)
    try:
        status = backend.poll(handle)
        while not status.is_terminal:
            if poll_interval:
                time.sleep(poll_interval)
            status = backend.poll(handle)

        if status is RunStatus.FAILED:
            detail = backend.error(handle) if hasattr(backend, "error") else ""
            raise GenerationFailed(f"La corrida {run_id} falló. {detail}".strip())

        artifacts = backend.fetch(handle, store.outputs_dir(run_id))
        store.write_metrics(run_id, artifacts.metrics)
    finally:
        backend.teardown(handle)

    return RunResult(run_id=run_id, artifacts=artifacts, cached=False)
