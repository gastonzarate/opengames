"""Interfaz que implementa cada lugar donde puede correr un modelo."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from core.job import Artifacts, Job, RunHandle, RunStatus


class BackendSpec(BaseModel):
    """Lo que el backend declara sobre sí mismo.

    `ephemeral` indica que el backend aprovisiona recursos facturables
    que `teardown()` debe liberar.
    """

    name: str
    vram_gb: int
    ephemeral: bool


@runtime_checkable
class Backend(Protocol):
    def capabilities(self) -> BackendSpec: ...

    def submit(self, job: Job) -> RunHandle: ...

    def poll(self, handle: RunHandle) -> RunStatus: ...

    def fetch(self, handle: RunHandle, dest: Path) -> Artifacts: ...

    def teardown(self, handle: RunHandle) -> None: ...
