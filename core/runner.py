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


# Piso de espera entre polls. Con poll_interval=0.0 (el default) esto evita
# que el bucle de `poll()` se convierta en un busy-loop a 100% de CPU si un
# backend real queda atascado en RUNNING. Cuando el llamador pasa un
# poll_interval mayor, se respeta ese valor en su lugar.
_MIN_POLL_INTERVAL = 0.01


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

    # Detectar colisiones de basename antes de copiar nada: dos inputs con
    # claves distintas pero el mismo nombre de archivo se pisarían en
    # silencio, y job.json/provenance.json quedarían diciendo que hubo dos
    # inputs cuando en disco quedó uno solo. Mismo criterio que
    # LocalBackend.fetch() usa para los outputs.
    basenames: dict[str, list[str]] = {}
    for name, source in job.inputs.items():
        basenames.setdefault(source.name, []).append(name)
    collisions = {bn: keys for bn, keys in basenames.items() if len(keys) > 1}
    if collisions:
        collision_msg = "; ".join(
            f"{bn} ({', '.join(sorted(keys))})" for bn, keys in sorted(collisions.items())
        )
        raise ValueError(
            f"Colisión de nombres de archivo entre los inputs del job: {collision_msg}. "
            f"Varios inputs no pueden compartir el mismo nombre de archivo."
        )
    for name, source in job.inputs.items():
        shutil.copy2(source, store.inputs_dir(run_id) / source.name)

    # `handle` arranca en None: si `submit()` lanza antes de devolver nada
    # (por ejemplo un backend real que aprovisiona un recurso pago a medias
    # y después falla), no hay nada que pasarle a `teardown()`. Mantener
    # `submit()` dentro del try garantiza que el `finally` corra siempre;
    # el guard de abajo evita llamar `teardown(None)`, que explotaría contra
    # cualquier implementación real que espere un RunHandle de verdad.
    handle = None
    try:
        handle = backend.submit(job)
        status = backend.poll(handle)
        while not status.is_terminal:
            time.sleep(max(poll_interval, _MIN_POLL_INTERVAL))
            status = backend.poll(handle)

        if status is RunStatus.FAILED:
            detail = backend.error(handle) if hasattr(backend, "error") else ""
            raise GenerationFailed(f"La corrida {run_id} falló. {detail}".strip())

        artifacts = backend.fetch(handle, store.outputs_dir(run_id))
        # El manifiesto de claves lógicas se persiste antes que las métricas:
        # así, si el proceso muere entre las dos escrituras, el marcador
        # `.in-progress` sigue presente y la corrida no queda cacheada a
        # medias (ver `RunStore.exists()`).
        store.write_artifacts(run_id, artifacts.files)
        store.write_metrics(run_id, artifacts.metrics)
    finally:
        if handle is not None:
            backend.teardown(handle)

    return RunResult(run_id=run_id, artifacts=artifacts, cached=False)
