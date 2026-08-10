"""Contrato entre los adapters de modelo y los backends de ejecución."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED)


class Job(BaseModel):
    """Lo que se pide. Serializable a JSON sin pérdida."""

    model: str
    inputs: dict[str, Path] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42


class Artifacts(BaseModel):
    """Lo que se obtiene."""

    files: dict[str, Path] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class RunHandle(BaseModel):
    """Referencia a una corrida en vuelo dentro de un backend."""

    backend: str
    run_id: str
    remote_id: str | None = None
