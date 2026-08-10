"""Ejecuta el adapter en el proceso actual. Sin aprovisionamiento."""

import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.backend import BackendSpec
from core.job import Artifacts, Job, RunHandle, RunStatus
from core.registry import get_model, register_backend


@dataclass
class _Run:
    status: RunStatus
    workdir: Path
    artifacts: Artifacts = field(default_factory=Artifacts)
    error: str = ""


@register_backend("local")
class LocalBackend:
    def __init__(self, vram_gb: int = 0, workroot: Path | None = None) -> None:
        self.vram_gb = vram_gb
        self.workroot = Path(workroot) if workroot else Path(tempfile.gettempdir()) / "opengames"
        self._runs: dict[str, _Run] = {}

    def capabilities(self) -> BackendSpec:
        return BackendSpec(name="local", vram_gb=self.vram_gb, ephemeral=False)

    def submit(self, job: Job) -> RunHandle:
        run_id = uuid.uuid4().hex[:12]
        workdir = self.workroot / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        record = _Run(status=RunStatus.RUNNING, workdir=workdir)
        self._runs[run_id] = record

        try:
            adapter = get_model(job.model)
            adapter.load()
            record.artifacts = adapter.generate(job, workdir)
            record.status = RunStatus.SUCCEEDED
        except Exception as exc:  # el backend reporta, no propaga
            record.status = RunStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"

        return RunHandle(backend="local", run_id=run_id, remote_id=run_id)

    def poll(self, handle: RunHandle) -> RunStatus:
        return self._runs[handle.run_id].status

    def error(self, handle: RunHandle) -> str:
        return self._runs[handle.run_id].error

    def fetch(self, handle: RunHandle, dest: Path) -> Artifacts:
        record = self._runs[handle.run_id]
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        copied = {}
        for name, source in record.artifacts.files.items():
            target = dest / Path(source).name
            shutil.copy2(source, target)
            copied[name] = target
        return Artifacts(files=copied, metrics=record.artifacts.metrics)

    def teardown(self, handle: RunHandle) -> None:
        record = self._runs.get(handle.run_id)
        if record is None:
            return
        shutil.rmtree(record.workdir, ignore_errors=True)
