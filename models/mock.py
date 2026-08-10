"""Adapter simulado. Ejercita el harness completo sin GPU ni pesos."""

import json
import struct
import time
from pathlib import Path

from core.job import Artifacts, Job
from core.model import Modality, ModelSpec
from core.registry import register_model

_GLB_MAGIC = 0x46546C67
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_UNSIGNED_SHORT = 5123
_FLOAT = 5126
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963


def _pad(data: bytes, filler: bytes) -> bytes:
    return data + filler * ((4 - len(data) % 4) % 4)


def minimal_glb() -> bytes:
    """Un triángulo, en un GLB binario válido según glTF 2.0."""
    indices = struct.pack("<3H", 0, 1, 2)
    positions = b"".join(
        struct.pack("<3f", *point)
        for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    )
    padded_indices = _pad(indices, b"\x00")
    binary = padded_indices + positions

    gltf = {
        "asset": {"version": "2.0", "generator": "opengames-mock"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 1}, "indices": 0}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": _UNSIGNED_SHORT,
                "count": 3,
                "type": "SCALAR",
            },
            {
                "bufferView": 1,
                "componentType": _FLOAT,
                "count": 3,
                "type": "VEC3",
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(indices),
                "target": _ELEMENT_ARRAY_BUFFER,
            },
            {
                "buffer": 0,
                "byteOffset": len(padded_indices),
                "byteLength": len(positions),
                "target": _ARRAY_BUFFER,
            },
        ],
        "buffers": [{"byteLength": len(binary)}],
    }

    json_chunk = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(binary)

    return (
        struct.pack("<III", _GLB_MAGIC, 2, total)
        + struct.pack("<II", len(json_chunk), _CHUNK_JSON)
        + json_chunk
        + struct.pack("<II", len(binary), _CHUNK_BIN)
        + binary
    )


@register_model("mock")
class MockModel:
    def describe(self) -> ModelSpec:
        return ModelSpec(
            name="mock",
            revision="0",
            min_vram_gb=0,
            accepts=[Modality.IMAGE, Modality.TEXT],
            produces=["glb"],
            docker_image="opengames/mock:0.1.0",
        )

    def load(self) -> None:
        return None

    def generate(self, job: Job, workdir: Path) -> Artifacts:
        started = time.perf_counter()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        target = workdir / "sample.glb"
        target.write_bytes(minimal_glb())
        return Artifacts(
            files={"sample.glb": target},
            metrics={"duration_s": time.perf_counter() - started},
        )
