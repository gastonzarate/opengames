"""Un experimento es el producto cartesiano de modelos, entradas y semillas."""

from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.job import Job
from core.registry import UnknownComponent, available_backends, get_backend_class, get_model
from core.runner import RunResult, execute
from core.runstore import RunStore


class InvalidBackendOptions(ValueError):
    """`backend_options` no coincide con el constructor del backend elegido."""


class ExperimentConfig(BaseModel):
    name: str
    backend: str
    backend_options: dict[str, Any] = Field(default_factory=dict)
    models: list[str]
    inputs: list[dict[str, Path]]
    params: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42])


def _resolve_relative_inputs(raw: Any, base_dir: Path) -> None:
    """Reescribe in-place las rutas relativas de `inputs` contra `base_dir`.

    La convención es la de docker-compose: las rutas de un config se
    interpretan relativas al archivo que las declara, no al directorio
    desde el que se invoca el CLI. Las rutas ya absolutas quedan intactas.
    """
    if not isinstance(raw, dict):
        return
    for entry in raw.get("inputs") or []:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            candidate = Path(value)
            if not candidate.is_absolute():
                entry[key] = str(base_dir / candidate)


def load_experiment(path: Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    _resolve_relative_inputs(raw, path.resolve().parent)
    config = ExperimentConfig.model_validate(raw)
    for name in config.models:  # falla temprano si el nombre no existe
        get_model(name)
    # Se comprueba por membresía, no instanciando: un backend puede exigir
    # argumentos de construcción que recién aparecen en `backend_options`.
    if config.backend not in available_backends():
        known = ", ".join(available_backends()) or "ninguno"
        raise UnknownComponent(f"No existe el backend '{config.backend}'. Registrados: {known}")
    return config


def expand_jobs(config: ExperimentConfig) -> list[Job]:
    return [
        Job(
            model=model,
            inputs=inputs,
            params=dict(config.params),
            export=dict(config.export),
            seed=seed,
        )
        for model, inputs, seed in product(config.models, config.inputs, config.seeds)
    ]


def run_experiment(config: ExperimentConfig, store: RunStore) -> list[RunResult]:
    backend_cls = get_backend_class(config.backend)
    try:
        backend = backend_cls(**config.backend_options)
    except TypeError as exc:
        raise InvalidBackendOptions(
            f"backend_options inválidas para el backend '{config.backend}' del "
            f"experimento '{config.name}': {exc}"
        ) from exc
    return [execute(job, backend, store) for job in expand_jobs(config)]
