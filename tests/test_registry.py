import pytest

from core import registry
from core.backend import BackendSpec
from core.model import Modality, ModelSpec


@pytest.fixture(autouse=True)
def clean_registry():
    registry.reset()
    yield
    registry.reset()


def _make_adapter():
    @registry.register_model("toy")
    class Toy:
        def describe(self):
            return ModelSpec(
                name="toy",
                revision="0",
                min_vram_gb=0,
                accepts=[Modality.IMAGE],
                produces=["glb"],
                docker_image="toy:0",
            )

        def load(self):
            pass

        def generate(self, job, workdir):
            raise NotImplementedError

    return Toy


def test_registered_model_is_retrievable_as_instance():
    _make_adapter()
    adapter = registry.get_model("toy")
    assert adapter.describe().name == "toy"
    assert "toy" in registry.available_models()


def test_unknown_model_lists_the_alternatives():
    _make_adapter()
    with pytest.raises(registry.UnknownComponent) as err:
        registry.get_model("nope")
    assert "toy" in str(err.value)


def test_duplicate_registration_is_rejected():
    _make_adapter()
    with pytest.raises(ValueError):
        _make_adapter()


def test_backends_use_a_separate_namespace():
    _make_adapter()

    @registry.register_backend("toy")
    class ToyBackend:
        def capabilities(self):
            return BackendSpec(name="toy", vram_gb=0, ephemeral=False)

        def submit(self, job):
            raise NotImplementedError

        def poll(self, handle):
            raise NotImplementedError

        def fetch(self, handle, dest):
            raise NotImplementedError

        def teardown(self, handle):
            pass

    assert registry.get_backend("toy").capabilities().vram_gb == 0
    assert registry.get_model("toy").describe().min_vram_gb == 0
