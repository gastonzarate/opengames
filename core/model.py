"""Interfaz que implementa cada modelo generativo."""

from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from core.job import Artifacts, Job


class Modality(str, Enum):
    IMAGE = "image"
    MULTIVIEW = "multiview"
    TEXT = "text"
    MESH = "mesh"


class ModelSpec(BaseModel):
    """Lo que el modelo declara sobre sí mismo.

    El runner usa `min_vram_gb` para rechazar combinaciones imposibles
    antes de aprovisionar hardware.
    """

    name: str
    revision: str
    min_vram_gb: int
    accepts: list[Modality]
    produces: list[str]
    docker_image: str


@runtime_checkable
class ModelAdapter(Protocol):
    def describe(self) -> ModelSpec: ...

    def load(self) -> None: ...

    def generate(self, job: Job, workdir: Path) -> Artifacts: ...
