import pytest

from core import registry
from core.job import Job, RunStatus


@pytest.fixture(autouse=True)
def wiring():
    import importlib

    import backends.local
    import models.mock

    # El reset va DESPUÉS del import: en la primera importación del proceso el
    # módulo ya se registra, y recargarlo sobre un registro no vacío choca con
    # la guardia de duplicados.
    registry.reset()
    importlib.reload(models.mock)
    importlib.reload(backends.local)
    yield
    registry.reset()


def test_capabilities_report_configured_vram():
    from backends.local import LocalBackend

    assert LocalBackend(vram_gb=8).capabilities().vram_gb == 8
    assert LocalBackend().capabilities().ephemeral is False


def test_submit_runs_the_adapter_and_fetch_returns_files(tmp_path):
    from backends.local import LocalBackend

    backend = LocalBackend(workroot=tmp_path / "work")
    handle = backend.submit(Job(model="mock"))

    assert backend.poll(handle) is RunStatus.SUCCEEDED

    dest = tmp_path / "out"
    artifacts = backend.fetch(handle, dest)
    assert artifacts.files["sample.glb"].parent == dest
    assert artifacts.files["sample.glb"].read_bytes()[:4] == b"glTF"


def test_failure_is_reported_not_raised(tmp_path):
    from core.model import Modality, ModelSpec

    @registry.register_model("explosive")
    class Explosive:
        def describe(self):
            return ModelSpec(
                name="explosive",
                revision="0",
                min_vram_gb=0,
                accepts=[Modality.IMAGE],
                produces=["glb"],
                docker_image="explosive:0",
            )

        def load(self):
            pass

        def generate(self, job, workdir):
            raise RuntimeError("boom")

    from backends.local import LocalBackend

    backend = LocalBackend(workroot=tmp_path / "work")
    handle = backend.submit(Job(model="explosive"))
    assert backend.poll(handle) is RunStatus.FAILED
    assert "boom" in backend.error(handle)


def test_teardown_is_idempotent(tmp_path):
    from backends.local import LocalBackend

    backend = LocalBackend(workroot=tmp_path / "work")
    handle = backend.submit(Job(model="mock"))
    backend.teardown(handle)
    backend.teardown(handle)
    assert backend.poll(handle) is RunStatus.SUCCEEDED


def test_fetch_detects_basename_collisions(tmp_path):
    from core.job import Artifacts
    from core.model import Modality, ModelSpec

    @registry.register_model("colliding")
    class CollidingAdapter:
        def describe(self):
            return ModelSpec(
                name="colliding",
                revision="0",
                min_vram_gb=0,
                accepts=[Modality.IMAGE],
                produces=["glb", "glb"],
                docker_image="colliding:0",
            )

        def load(self):
            pass

        def generate(self, job, workdir):
            # Crear artefactos con el mismo basename en subdirectorios distintos
            (workdir / "views" / "front").mkdir(parents=True)
            (workdir / "views" / "back").mkdir(parents=True)
            front_file = workdir / "views" / "front" / "render.glb"
            back_file = workdir / "views" / "back" / "render.glb"
            front_file.write_bytes(b"glTF")
            back_file.write_bytes(b"glTF")
            return Artifacts(files={"front": front_file, "back": back_file})

    from backends.local import LocalBackend

    backend = LocalBackend(workroot=tmp_path / "work")
    handle = backend.submit(Job(model="colliding"))
    assert backend.poll(handle) is RunStatus.SUCCEEDED

    dest = tmp_path / "out"
    with pytest.raises(ValueError) as exc_info:
        backend.fetch(handle, dest)
    assert "basename collisions" in str(exc_info.value).lower()
    assert "render.glb" in str(exc_info.value)
    assert "front" in str(exc_info.value)
    assert "back" in str(exc_info.value)


def test_fetch_after_teardown_fails_clearly(tmp_path):
    from backends.local import LocalBackend

    backend = LocalBackend(workroot=tmp_path / "work")
    handle = backend.submit(Job(model="mock"))
    backend.teardown(handle)

    dest = tmp_path / "out"
    with pytest.raises(RuntimeError) as exc_info:
        backend.fetch(handle, dest)
    assert "torn down" in str(exc_info.value).lower()
    assert "no longer available" in str(exc_info.value).lower()
