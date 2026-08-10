import pytest

from core import registry
from core.job import Job
from core.runstore import RunStore


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


def test_execute_produces_a_run_directory(tmp_path):
    from backends.local import LocalBackend
    from core.runner import execute

    store = RunStore(tmp_path / "runs")
    result = execute(Job(model="mock"), LocalBackend(workroot=tmp_path / "w"), store)

    assert result.cached is False
    assert store.exists(result.run_id)
    base = tmp_path / "runs" / result.run_id
    assert (base / "job.json").is_file()
    assert (base / "provenance.json").is_file()
    assert (base / "outputs" / "sample.glb").is_file()


def test_second_identical_execute_hits_the_cache(tmp_path):
    from backends.local import LocalBackend
    from core.runner import execute

    store = RunStore(tmp_path / "runs")
    job = Job(model="mock")
    first = execute(job, LocalBackend(workroot=tmp_path / "w"), store)
    second = execute(job, LocalBackend(workroot=tmp_path / "w"), store)

    assert second.run_id == first.run_id
    assert second.cached is True


def test_insufficient_vram_fails_before_submitting(tmp_path):
    from core.job import Artifacts, RunHandle, RunStatus
    from core.model import Modality, ModelSpec
    from core.backend import BackendSpec
    from core.runner import InsufficientVram, execute

    @registry.register_model("hungry")
    class Hungry:
        def describe(self):
            return ModelSpec(
                name="hungry",
                revision="0",
                min_vram_gb=24,
                accepts=[Modality.IMAGE],
                produces=["glb"],
                docker_image="hungry:0",
            )

        def load(self):
            pass

        def generate(self, job, workdir):
            raise AssertionError("no debería ejecutarse")

    class Tiny:
        submitted = False

        def capabilities(self):
            return BackendSpec(name="tiny", vram_gb=8, ephemeral=False)

        def submit(self, job):
            Tiny.submitted = True
            return RunHandle(backend="tiny", run_id="x")

        def poll(self, handle):
            return RunStatus.SUCCEEDED

        def fetch(self, handle, dest):
            return Artifacts()

        def teardown(self, handle):
            pass

    with pytest.raises(InsufficientVram) as err:
        execute(Job(model="hungry"), Tiny(), RunStore(tmp_path / "runs"))

    assert "24" in str(err.value) and "8" in str(err.value)
    assert Tiny.submitted is False


def test_teardown_runs_even_when_generation_fails(tmp_path):
    from core.model import Modality, ModelSpec
    from core.runner import GenerationFailed, execute
    from backends.local import LocalBackend

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

    backend = LocalBackend(workroot=tmp_path / "w")
    torn: list[str] = []
    original = backend.teardown
    backend.teardown = lambda handle: (torn.append(handle.run_id), original(handle))[1]

    with pytest.raises(GenerationFailed):
        execute(Job(model="explosive"), backend, RunStore(tmp_path / "runs"))

    assert len(torn) == 1


def test_failed_run_is_not_cached(tmp_path):
    from core.model import Modality, ModelSpec
    from core.runner import GenerationFailed, execute
    from backends.local import LocalBackend

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

    store = RunStore(tmp_path / "runs")
    job = Job(model="explosive")
    for _ in range(2):
        with pytest.raises(GenerationFailed):
            execute(job, LocalBackend(workroot=tmp_path / "w"), store)
