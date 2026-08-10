import json
import struct

import pytest

from core import registry
from core.job import Job
from core.model import Modality


@pytest.fixture(autouse=True)
def load_models():
    import importlib

    import models.mock

    # El reset va DESPUÉS del import: en la primera importación del proceso el
    # módulo ya se registra, y recargarlo sobre un registro no vacío choca con
    # la guardia de duplicados.
    registry.reset()
    importlib.reload(models.mock)
    yield
    registry.reset()


def _parse_glb(raw: bytes) -> dict:
    magic, version, total = struct.unpack("<III", raw[:12])
    assert magic == 0x46546C67, "cabecera glTF inválida"
    assert version == 2
    assert total == len(raw), "el tamaño declarado no coincide con el archivo"
    offset, gltf = 12, None
    while offset < len(raw):
        length, kind = struct.unpack("<II", raw[offset : offset + 8])
        chunk = raw[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf8"))
        offset += 8 + length + ((4 - length % 4) % 4)
    assert gltf is not None, "no hay chunk JSON"
    return gltf


def test_minimal_glb_is_parseable():
    from models.mock import minimal_glb

    gltf = _parse_glb(minimal_glb())
    assert gltf["asset"]["version"] == "2.0"
    assert len(gltf["meshes"]) == 1
    assert gltf["accessors"][0]["count"] == 3


def test_mock_declares_no_vram_requirement():
    spec = registry.get_model("mock").describe()
    assert spec.min_vram_gb == 0
    assert Modality.IMAGE in spec.accepts
    assert "glb" in spec.produces


def test_generate_writes_a_glb_and_reports_duration(tmp_path):
    adapter = registry.get_model("mock")
    adapter.load()
    artifacts = adapter.generate(Job(model="mock"), tmp_path)

    glb = artifacts.files["sample.glb"]
    assert glb.parent == tmp_path
    _parse_glb(glb.read_bytes())
    assert artifacts.metrics["duration_s"] >= 0.0


def test_generate_is_deterministic_for_a_given_seed(tmp_path):
    adapter = registry.get_model("mock")
    a = adapter.generate(Job(model="mock", seed=1), tmp_path / "a")
    b = adapter.generate(Job(model="mock", seed=1), tmp_path / "b")
    assert a.files["sample.glb"].read_bytes() == b.files["sample.glb"].read_bytes()
