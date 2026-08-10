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

    def submit(self, job: Job) -> RunHandle:
        """Arranca la corrida y devuelve un handle para seguirla con `poll()`.

        Contrato de aprovisionamiento: si `submit()` lanza, no debe dejar
        ningún recurso facturable a medio aprovisionar sin liberar. El
        runner (`core/runner.py::execute`) solo llama a `teardown()` cuando
        `submit()` devolvió un handle; si falla antes de devolver nada, no
        hay handle que pasarle y nadie más va a liberar lo que haya
        alcanzado a reservar. Un backend real (RunPod, EC2, SageMaker) que
        levante una instancia o una GPU y después falle -por ejemplo, al
        subir el job o configurar la red- tiene que liberar esa reserva él
        mismo, dentro de `submit()`, antes de relanzar la excepción.
        Dejarlo sin liberar es una GPU paga encendida que nadie apaga.
        """
        ...

    def poll(self, handle: RunHandle) -> RunStatus: ...

    def error(self, handle: RunHandle) -> str:
        """Detalle legible del fallo cuando `poll()` devolvió `RunStatus.FAILED`.

        Parte del contrato del backend: el runner lo llama para incluir el
        motivo en `GenerationFailed`. Devolver cadena vacía si no hay más
        detalle que dar.
        """
        ...

    def fetch(self, handle: RunHandle, dest: Path) -> Artifacts: ...

    def teardown(self, handle: RunHandle) -> None: ...
