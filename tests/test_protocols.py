from pathlib import Path

from core.backend import Backend, BackendSpec
from core.job import Artifacts, Job, RunHandle, RunStatus
from core.model import Modality, ModelAdapter, ModelSpec


def test_model_spec_declares_requirements():
    spec = ModelSpec(
        name="trellis2",
        revision="main",
        min_vram_gb=24,
        accepts=[Modality.IMAGE],
        produces=["glb", "preview"],
        docker_image="opengames/trellis2:0.1.0",
    )
    assert spec.min_vram_gb == 24
    assert Modality.IMAGE in spec.accepts


def test_model_spec_rejects_unknown_modality():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        ModelSpec(
            name="x",
            revision="main",
            min_vram_gb=1,
            accepts=["hologram"],
            produces=["glb"],
            docker_image="x:1",
        )


def test_adapter_duck_types_against_protocol():
    class Dummy:
        def describe(self) -> ModelSpec:
            return ModelSpec(
                name="dummy",
                revision="0",
                min_vram_gb=0,
                accepts=[Modality.IMAGE],
                produces=["glb"],
                docker_image="dummy:0",
            )

        def load(self) -> None:
            pass

        def generate(self, job: Job, workdir: Path) -> Artifacts:
            return Artifacts()

    assert isinstance(Dummy(), ModelAdapter)


def test_backend_duck_types_against_protocol():
    class Dummy:
        def capabilities(self) -> BackendSpec:
            return BackendSpec(name="dummy", vram_gb=0, ephemeral=False)

        def submit(self, job: Job) -> RunHandle:
            return RunHandle(backend="dummy", run_id="0")

        def poll(self, handle: RunHandle) -> RunStatus:
            return RunStatus.SUCCEEDED

        def fetch(self, handle: RunHandle, dest: Path) -> Artifacts:
            return Artifacts()

        def teardown(self, handle: RunHandle) -> None:
            pass

    assert isinstance(Dummy(), Backend)


def test_incomplete_adapter_fails_protocol_check():
    class Broken:
        def describe(self) -> ModelSpec: ...

    assert not isinstance(Broken(), ModelAdapter)
