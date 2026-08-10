"""Un experimento es el producto cartesiano de modelos, entradas y semillas."""

from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.job import Job
from core.registry import UnknownComponent, available_backends, get_backend, get_model
from core.runner import RunResult, execute
from core.runstore import RunStore


class ExperimentConfig(BaseModel):
    name: str
    backend: str
    backend_options: dict[str, Any] = Field(default_factory=dict)
    models: list[str]
    inputs: list[dict[str, Path]]
    params: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42])


def load_experiment(path: Path) -> ExperimentConfig:
    config = ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
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
    backend_cls = type(get_backend(config.backend))
    backend = backend_cls(**config.backend_options)
    return [execute(job, backend, store) for job in expand_jobs(config)]
