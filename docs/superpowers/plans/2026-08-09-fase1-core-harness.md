# Fase 1 — Núcleo del harness de experimentación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el núcleo que desacopla *qué modelo* de *dónde corre*, ejecutable de punta a punta con un adapter simulado y sin GPU.

**Architecture:** Dos protocolos independientes —`ModelAdapter` y `Backend`— unidos por un contrato `Job` serializable a JSON. Un registro resuelve nombres de config a implementaciones. Un almacén de corridas direcciona por contenido para cachear resultados y guarda la procedencia de cada ejecución. El runner valida VRAM antes de aprovisionar y garantiza el `teardown`.

**Tech Stack:** Python 3.11, pydantic v2, PyYAML, pytest. Sin dependencias de nube en esta fase.

## Global Constraints

- **Python 3.11 o superior** para el núcleo. Los adapters corren en sus propios contenedores y no comparten este piso de versión.
- **`models/` no puede importar SDKs de nube.** Un `import boto3` o `import runpod` dentro de `models/` es un error de diseño: el transporte es responsabilidad de `backends/`.
- **`backends/` no puede importar de `models/`.** La única dependencia permitida entre ambos es `core/`.
- **El `run_id` no puede depender del tiempo, del azar ni de rutas absolutas.** Solo de contenido: modelo, revisión, parámetros, semilla y hash de los archivos de entrada.
- **`teardown()` es idempotente** y debe ejecutarse aunque la corrida falle.
- **Solo modelos con licencia MIT o Apache-2.0.**
- Todo el código de esta fase corre en CI sin GPU.

---

### Task 1: Andamiaje del proyecto y contrato `Job`

**Files:**
- Create: `pyproject.toml`
- Create: `core/__init__.py`
- Create: `core/job.py`
- Create: `tests/__init__.py`
- Create: `tests/test_job.py`

**Interfaces:**
- Consumes: nada.
- Produces: `RunStatus` (enum de str), `Job(model: str, inputs: dict[str, Path], params: dict[str, Any], export: dict[str, Any], seed: int)`, `Artifacts(files: dict[str, Path], metrics: dict[str, float])`, `RunHandle(backend: str, run_id: str, remote_id: str | None)`.

- [ ] **Step 1: Crear `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "opengames"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pydantic>=2.6", "PyYAML>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["core*", "models*", "backends*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Crear el entorno e instalar**

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

- [ ] **Step 3: Escribir el test que falla**

Crear `tests/__init__.py` vacío y `tests/test_job.py`:

```python
import json
from pathlib import Path

from core.job import Artifacts, Job, RunHandle, RunStatus


def test_job_has_defaults():
    job = Job(model="mock")
    assert job.seed == 42
    assert job.inputs == {}
    assert job.params == {}
    assert job.export == {}


def test_job_roundtrips_through_json():
    job = Job(
        model="trellis2",
        inputs={"image": Path("assets/ref.png")},
        params={"pipeline_type": "512"},
        export={"texture_size": 2048},
        seed=7,
    )
    restored = Job.model_validate(json.loads(job.model_dump_json()))
    assert restored == job


def test_artifacts_defaults_are_independent():
    a, b = Artifacts(), Artifacts()
    a.files["glb"] = Path("out.glb")
    assert b.files == {}


def test_run_status_values():
    assert RunStatus.SUCCEEDED.value == "succeeded"
    assert {s.value for s in RunStatus} == {
        "pending",
        "running",
        "succeeded",
        "failed",
    }


def test_run_handle_remote_id_optional():
    handle = RunHandle(backend="local", run_id="abc123")
    assert handle.remote_id is None
```

- [ ] **Step 4: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_job.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core'`

- [ ] **Step 5: Implementar `core/job.py`**

Crear `core/__init__.py` vacío y `core/job.py`:

```python
"""Contrato entre los adapters de modelo y los backends de ejecución."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.SUCCEEDED, RunStatus.FAILED)


class Job(BaseModel):
    """Lo que se pide. Serializable a JSON sin pérdida."""

    model: str
    inputs: dict[str, Path] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42


class Artifacts(BaseModel):
    """Lo que se obtiene."""

    files: dict[str, Path] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


class RunHandle(BaseModel):
    """Referencia a una corrida en vuelo dentro de un backend."""

    backend: str
    run_id: str
    remote_id: str | None = None
```

- [ ] **Step 6: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_job.py -v`
Expected: PASS, 5 tests

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml core/ tests/
git commit -m "feat(core): contrato Job, Artifacts y RunHandle"
```

---

### Task 2: Protocolos `ModelAdapter` y `Backend`

**Files:**
- Create: `core/model.py`
- Create: `core/backend.py`
- Create: `tests/test_protocols.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts`, `RunHandle`, `RunStatus` de `core.job`.
- Produces: `Modality` (enum), `ModelSpec(name, revision, min_vram_gb, accepts, produces, docker_image)`, `ModelAdapter` (Protocol con `describe()`, `load()`, `generate(job, workdir)`), `BackendSpec(name, vram_gb, ephemeral)`, `Backend` (Protocol con `capabilities()`, `submit(job)`, `poll(handle)`, `fetch(handle, dest)`, `teardown(handle)`).

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_protocols.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core.model'`

- [ ] **Step 3: Implementar `core/model.py`**

```python
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
```

- [ ] **Step 4: Implementar `core/backend.py`**

```python
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
```

- [ ] **Step 5: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_protocols.py -v`
Expected: PASS, 5 tests

- [ ] **Step 6: Commit**

```bash
git add core/model.py core/backend.py tests/test_protocols.py
git commit -m "feat(core): protocolos ModelAdapter y Backend"
```

---

### Task 3: Registro de nombres a implementaciones

**Files:**
- Create: `core/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `ModelAdapter` de `core.model`, `Backend` de `core.backend`.
- Produces: `register_model(name)` y `register_backend(name)` (decoradores de clase), `get_model(name) -> ModelAdapter`, `get_backend(name) -> Backend`, `available_models() -> list[str]`, `available_backends() -> list[str]`, `UnknownComponent(KeyError)`.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_registry.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core.registry'`

- [ ] **Step 3: Implementar `core/registry.py`**

```python
"""Resuelve los nombres que aparecen en los configs a implementaciones."""

from typing import Callable, TypeVar

from core.backend import Backend
from core.model import ModelAdapter

T = TypeVar("T")

_MODELS: dict[str, type] = {}
_BACKENDS: dict[str, type] = {}


class UnknownComponent(KeyError):
    """El nombre del config no corresponde a nada registrado."""


def _register(store: dict[str, type], kind: str, name: str) -> Callable[[type[T]], type[T]]:
    def decorator(cls: type[T]) -> type[T]:
        if name in store:
            raise ValueError(f"Ya hay un {kind} registrado como '{name}'")
        store[name] = cls
        return cls

    return decorator


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_MODELS, "modelo", name)


def register_backend(name: str) -> Callable[[type[T]], type[T]]:
    return _register(_BACKENDS, "backend", name)


def _get(store: dict[str, type], kind: str, name: str):
    if name not in store:
        known = ", ".join(sorted(store)) or "ninguno"
        raise UnknownComponent(f"No existe el {kind} '{name}'. Registrados: {known}")
    return store[name]()


def get_model(name: str) -> ModelAdapter:
    return _get(_MODELS, "modelo", name)


def get_backend(name: str) -> Backend:
    return _get(_BACKENDS, "backend", name)


def available_models() -> list[str]:
    return sorted(_MODELS)


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def reset() -> None:
    """Solo para tests: vacía ambos espacios de nombres."""
    _MODELS.clear()
    _BACKENDS.clear()
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add core/registry.py tests/test_registry.py
git commit -m "feat(core): registro de modelos y backends"
```

---

### Task 4: Almacén de corridas con direccionamiento por contenido

**Files:**
- Create: `core/runstore.py`
- Create: `tests/test_runstore.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts` de `core.job`; `ModelSpec` de `core.model`.
- Produces: `compute_run_id(job, spec) -> str` (16 hex), `collect_provenance(job, spec, backend_name) -> dict`, `RunStore(root: Path)` con `exists(run_id) -> bool`, `create(run_id) -> Path`, `inputs_dir(run_id)`, `outputs_dir(run_id)`, `write_job(run_id, job)`, `write_provenance(run_id, data)`, `write_metrics(run_id, metrics)`, `load_artifacts(run_id) -> Artifacts`.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_runstore.py`:

```python
import json

import pytest

from core.job import Artifacts, Job
from core.model import Modality, ModelSpec
from core.runstore import RunStore, collect_provenance, compute_run_id


@pytest.fixture
def spec():
    return ModelSpec(
        name="toy",
        revision="rev1",
        min_vram_gb=0,
        accepts=[Modality.IMAGE],
        produces=["glb"],
        docker_image="toy:0",
    )


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "ref.png"
    p.write_bytes(b"contenido-de-la-imagen")
    return p


def test_run_id_is_stable_across_calls(spec, image):
    job = Job(model="toy", inputs={"image": image})
    assert compute_run_id(job, spec) == compute_run_id(job, spec)


def test_run_id_ignores_the_input_path_but_not_its_content(spec, image, tmp_path):
    twin = tmp_path / "otro-nombre.png"
    twin.write_bytes(image.read_bytes())
    a = compute_run_id(Job(model="toy", inputs={"image": image}), spec)
    b = compute_run_id(Job(model="toy", inputs={"image": twin}), spec)
    assert a == b

    changed = tmp_path / "distinta.png"
    changed.write_bytes(b"otro contenido")
    c = compute_run_id(Job(model="toy", inputs={"image": changed}), spec)
    assert c != a


def test_run_id_changes_with_params_seed_and_revision(spec, image):
    base = Job(model="toy", inputs={"image": image})
    assert compute_run_id(base.model_copy(update={"seed": 1}), spec) != compute_run_id(base, spec)
    assert compute_run_id(
        base.model_copy(update={"params": {"pipeline_type": "1024"}}), spec
    ) != compute_run_id(base, spec)
    assert compute_run_id(base, spec.model_copy(update={"revision": "rev2"})) != compute_run_id(
        base, spec
    )


def test_run_id_ignores_param_ordering(spec, image):
    a = Job(model="toy", inputs={"image": image}, params={"a": 1, "b": 2})
    b = Job(model="toy", inputs={"image": image}, params={"b": 2, "a": 1})
    assert compute_run_id(a, spec) == compute_run_id(b, spec)


def test_provenance_carries_the_docker_digest(spec, image):
    job = Job(model="toy", inputs={"image": image}, seed=9)
    prov = collect_provenance(job, spec, backend_name="local")
    assert prov["seed"] == 9
    assert prov["model_revision"] == "rev1"
    assert prov["docker_image"] == "toy:0"
    assert prov["backend"] == "local"
    assert "repo_sha" in prov
    assert "created_at" in prov


def test_store_writes_and_reads_a_run(tmp_path, spec, image):
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    assert not store.exists(run_id)
    store.create(run_id)
    store.write_job(run_id, job)
    store.write_provenance(run_id, collect_provenance(job, spec, "local"))
    (store.outputs_dir(run_id) / "sample.glb").write_bytes(b"glb")
    store.write_metrics(run_id, {"duration_s": 1.5})

    assert store.exists(run_id)
    loaded = store.load_artifacts(run_id)
    assert loaded.metrics["duration_s"] == 1.5
    assert loaded.files["sample.glb"].read_bytes() == b"glb"

    written = json.loads((store.create(run_id) / "job.json").read_text())
    assert written["model"] == "toy"


def test_exists_is_false_until_metrics_are_written(tmp_path, spec, image):
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)
    store.create(run_id)
    store.write_job(run_id, job)
    assert not store.exists(run_id)
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_runstore.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core.runstore'`

- [ ] **Step 3: Implementar `core/runstore.py`**

```python
"""Persistencia de corridas: identidad por contenido, procedencia y caché.

`exists()` solo devuelve verdadero cuando existe `metrics.json`, que se
escribe al final. Una corrida interrumpida deja el directorio a medias y
no se toma como cacheada.
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.job import Artifacts, Job
from core.model import ModelSpec

_CHUNK = 1 << 20


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()


def compute_run_id(job: Job, spec: ModelSpec) -> str:
    """Identidad por contenido. No depende del tiempo ni de rutas."""
    payload = {
        "model": spec.name,
        "revision": spec.revision,
        "params": job.params,
        "export": job.export,
        "seed": job.seed,
        "inputs": {key: _hash_file(path) for key, path in sorted(job.inputs.items())},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:16]


def _repo_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def collect_provenance(job: Job, spec: ModelSpec, backend_name: str) -> dict[str, Any]:
    return {
        "model": spec.name,
        "model_revision": spec.revision,
        "docker_image": spec.docker_image,
        "backend": backend_name,
        "seed": job.seed,
        "repo_sha": _repo_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, run_id: str) -> Path:
        return self.root / run_id

    def create(self, run_id: str) -> Path:
        base = self._dir(run_id)
        (base / "inputs").mkdir(parents=True, exist_ok=True)
        (base / "outputs").mkdir(parents=True, exist_ok=True)
        return base

    def exists(self, run_id: str) -> bool:
        return (self._dir(run_id) / "metrics.json").is_file()

    def inputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "inputs"

    def outputs_dir(self, run_id: str) -> Path:
        return self.create(run_id) / "outputs"

    def write_job(self, run_id: str, job: Job) -> None:
        (self.create(run_id) / "job.json").write_text(job.model_dump_json(indent=2))

    def write_provenance(self, run_id: str, data: dict[str, Any]) -> None:
        (self.create(run_id) / "provenance.json").write_text(json.dumps(data, indent=2))

    def write_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        (self.create(run_id) / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def load_artifacts(self, run_id: str) -> Artifacts:
        metrics_path = self._dir(run_id) / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
        outputs = self.outputs_dir(run_id)
        files = {path.name: path for path in sorted(outputs.iterdir()) if path.is_file()}
        return Artifacts(files=files, metrics=metrics)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_runstore.py -v`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add core/runstore.py tests/test_runstore.py
git commit -m "feat(core): almacén de corridas con id por contenido y procedencia"
```

---

### Task 5: Adapter simulado que produce un GLB válido

**Files:**
- Create: `models/__init__.py`
- Create: `models/mock.py`
- Create: `tests/test_mock_model.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts` de `core.job`; `Modality`, `ModelSpec` de `core.model`; `register_model` de `core.registry`.
- Produces: `minimal_glb() -> bytes`, clase `MockModel` registrada como `"mock"`.

El GLB tiene que ser válido de verdad: la etapa `evaluate` de la fase 3 lo va a parsear, y un archivo de mentira haría pasar tests que después fallan con datos reales.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_mock_model.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_mock_model.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'models'`

- [ ] **Step 3: Implementar `models/mock.py`**

Crear `models/__init__.py` vacío y `models/mock.py`:

```python
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
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_mock_model.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add models/ tests/test_mock_model.py
git commit -m "feat(models): adapter simulado que emite un GLB válido"
```

---

### Task 6: Backend local

**Files:**
- Create: `backends/__init__.py`
- Create: `backends/local.py`
- Create: `tests/test_local_backend.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts`, `RunHandle`, `RunStatus` de `core.job`; `BackendSpec` de `core.backend`; `get_model` de `core.registry`.
- Produces: clase `LocalBackend` registrada como `"local"`, con constructor `LocalBackend(vram_gb: int = 0, workroot: Path | None = None)`.

`LocalBackend` ejecuta el adapter en el mismo proceso. `submit()` corre la generación de forma sincrónica y guarda el resultado en memoria; `poll()` devuelve el estado ya conocido. Es deliberadamente el backend más simple: sirve de referencia para los remotos y de banco de pruebas del runner.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_local_backend.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_local_backend.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'backends'`

- [ ] **Step 3: Implementar `backends/local.py`**

Crear `backends/__init__.py` vacío y `backends/local.py`:

```python
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
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_local_backend.py -v`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add backends/ tests/test_local_backend.py
git commit -m "feat(backends): backend local en proceso"
```

---

### Task 7: Runner con validación de VRAM, caché y teardown garantizado

**Files:**
- Create: `core/runner.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: todo lo anterior de `core`.
- Produces: `InsufficientVram(RuntimeError)`, `GenerationFailed(RuntimeError)`, `RunResult(run_id: str, artifacts: Artifacts, cached: bool)`, `execute(job, backend, store, poll_interval=0.0) -> RunResult`.

`execute` recibe la **instancia** del backend, no su nombre, para que quien llama controle su configuración (por ejemplo la VRAM de `local`). La resolución por nombre ocurre en la capa de config de la Task 8.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_runner.py`:

```python
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
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_runner.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core.runner'`

- [ ] **Step 3: Implementar `core/runner.py`**

```python
"""Orquesta una corrida: valida, cachea, ejecuta y limpia."""

import shutil
import time
from dataclasses import dataclass

from core.backend import Backend
from core.job import Artifacts, Job, RunStatus
from core.registry import get_model
from core.runstore import RunStore, collect_provenance, compute_run_id


class InsufficientVram(RuntimeError):
    """El backend elegido no llega a la VRAM que el modelo declara."""


class GenerationFailed(RuntimeError):
    """El backend reportó estado FAILED."""


@dataclass
class RunResult:
    run_id: str
    artifacts: Artifacts
    cached: bool


def execute(
    job: Job,
    backend: Backend,
    store: RunStore,
    poll_interval: float = 0.0,
) -> RunResult:
    spec = get_model(job.model).describe()
    caps = backend.capabilities()

    if spec.min_vram_gb > caps.vram_gb:
        raise InsufficientVram(
            f"El modelo '{spec.name}' necesita {spec.min_vram_gb} GB de VRAM y el "
            f"backend '{caps.name}' ofrece {caps.vram_gb} GB. "
            f"Elegí un backend con más memoria o un modelo más chico."
        )

    run_id = compute_run_id(job, spec)
    if store.exists(run_id):
        return RunResult(run_id=run_id, artifacts=store.load_artifacts(run_id), cached=True)

    store.create(run_id)
    store.write_job(run_id, job)
    store.write_provenance(run_id, collect_provenance(job, spec, caps.name))
    for name, source in job.inputs.items():
        shutil.copy2(source, store.inputs_dir(run_id) / source.name)

    handle = backend.submit(job)
    try:
        status = backend.poll(handle)
        while not status.is_terminal:
            if poll_interval:
                time.sleep(poll_interval)
            status = backend.poll(handle)

        if status is RunStatus.FAILED:
            detail = backend.error(handle) if hasattr(backend, "error") else ""
            raise GenerationFailed(f"La corrida {run_id} falló. {detail}".strip())

        artifacts = backend.fetch(handle, store.outputs_dir(run_id))
        store.write_metrics(run_id, artifacts.metrics)
    finally:
        backend.teardown(handle)

    return RunResult(run_id=run_id, artifacts=artifacts, cached=False)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_runner.py -v`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add core/runner.py tests/test_runner.py
git commit -m "feat(core): runner con validación de VRAM, caché y teardown garantizado"
```

---

### Task 8: Config declarativo de experimentos y CLI

**Files:**
- Create: `core/experiment.py`
- Create: `cli.py`
- Create: `experiments/smoke.yaml`
- Create: `tests/test_experiment.py`

**Interfaces:**
- Consumes: todo `core`, más `backends.local` y `models.mock` para el registro.
- Produces: `ExperimentConfig(name, backend, backend_options, models, inputs, params, export, seeds)`, `load_experiment(path) -> ExperimentConfig`, `expand_jobs(config) -> list[Job]`, `run_experiment(config, store) -> list[RunResult]`.

Un experimento cruza modelos × entradas × semillas. Esa expansión es lo que hace que comparar dos modelos sea un archivo YAML y no código nuevo.

- [ ] **Step 1: Escribir el test que falla**

Crear `tests/test_experiment.py`:

```python
import pytest
import yaml

from core import registry
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


def _write_config(tmp_path, **overrides):
    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_a.write_bytes(b"aaa")
    image_b.write_bytes(b"bbb")
    payload = {
        "name": "smoke",
        "backend": "local",
        "backend_options": {"vram_gb": 0},
        "models": ["mock"],
        "inputs": [{"image": str(image_a)}, {"image": str(image_b)}],
        "params": {"pipeline_type": "512"},
        "export": {"texture_size": 2048},
        "seeds": [1, 2],
    }
    payload.update(overrides)
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def test_expand_produces_the_full_cross_product(tmp_path):
    from core.experiment import expand_jobs, load_experiment

    config = load_experiment(_write_config(tmp_path))
    jobs = expand_jobs(config)

    assert len(jobs) == 4  # 1 modelo × 2 entradas × 2 semillas
    assert {job.seed for job in jobs} == {1, 2}
    assert all(job.params["pipeline_type"] == "512" for job in jobs)
    assert all(job.export["texture_size"] == 2048 for job in jobs)


def test_unknown_model_in_config_is_rejected_at_load(tmp_path):
    from core.experiment import load_experiment
    from core.registry import UnknownComponent

    with pytest.raises(UnknownComponent):
        load_experiment(_write_config(tmp_path, models=["inexistente"]))


def test_run_experiment_executes_every_job(tmp_path):
    from core.experiment import load_experiment, run_experiment

    store = RunStore(tmp_path / "runs")
    results = run_experiment(load_experiment(_write_config(tmp_path)), store)

    assert len(results) == 4
    assert len({r.run_id for r in results}) == 4
    assert all(store.exists(r.run_id) for r in results)


def test_rerunning_an_experiment_is_fully_cached(tmp_path):
    from core.experiment import load_experiment, run_experiment

    store = RunStore(tmp_path / "runs")
    config = load_experiment(_write_config(tmp_path))
    run_experiment(config, store)
    second = run_experiment(config, store)

    assert all(r.cached for r in second)
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_experiment.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'core.experiment'`

- [ ] **Step 3: Implementar `core/experiment.py`**

```python
"""Un experimento es el producto cartesiano de modelos, entradas y semillas."""

from itertools import product
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from core.job import Job
from core.registry import UnknownComponent, available_backends, get_backend, get_model
from core.runner import RunResult, execute
from core.runstore import RunStore


class ExperimentConfig(BaseModel):
    name: str
    backend: str
    backend_options: dict[str, Any] = Field(default_factory=dict)
    models: list[str]
    inputs: list[dict[str, Path]]
    params: dict[str, Any] = Field(default_factory=dict)
    export: dict[str, Any] = Field(default_factory=dict)
    seeds: list[int] = Field(default_factory=lambda: [42])


def load_experiment(path: Path) -> ExperimentConfig:
    config = ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))
    for name in config.models:  # falla temprano si el nombre no existe
        get_model(name)
    # Se comprueba por membresía, no instanciando: un backend puede exigir
    # argumentos de construcción que recién aparecen en `backend_options`.
    if config.backend not in available_backends():
        known = ", ".join(available_backends()) or "ninguno"
        raise UnknownComponent(f"No existe el backend '{config.backend}'. Registrados: {known}")
    return config


def expand_jobs(config: ExperimentConfig) -> list[Job]:
    return [
        Job(
            model=model,
            inputs=inputs,
            params=dict(config.params),
            export=dict(config.export),
            seed=seed,
        )
        for model, inputs, seed in product(config.models, config.inputs, config.seeds)
    ]


def run_experiment(config: ExperimentConfig, store: RunStore) -> list[RunResult]:
    backend_cls = type(get_backend(config.backend))
    backend = backend_cls(**config.backend_options)
    return [execute(job, backend, store) for job in expand_jobs(config)]
```

- [ ] **Step 4: Implementar `cli.py`**

```python
"""Punto de entrada: `python cli.py run experiments/smoke.yaml`."""

import argparse
import sys
from pathlib import Path

import backends.local  # noqa: F401  registra el backend
import models.mock  # noqa: F401  registra el modelo
from core.experiment import load_experiment, run_experiment
from core.registry import available_backends, available_models
from core.runstore import RunStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="opengames")
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Ejecuta un experimento")
    run_cmd.add_argument("config", type=Path)
    run_cmd.add_argument("--runs-dir", type=Path, default=Path("runs"))

    sub.add_parser("list", help="Muestra modelos y backends registrados")

    args = parser.parse_args(argv)

    if args.command == "list":
        print("Modelos: ", ", ".join(available_models()))
        print("Backends:", ", ".join(available_backends()))
        return 0

    results = run_experiment(load_experiment(args.config), RunStore(args.runs_dir))
    for result in results:
        marca = "cache" if result.cached else "nuevo"
        print(f"[{marca}] {result.run_id}")
    print(f"{len(results)} corridas, {sum(r.cached for r in results)} desde caché")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Crear `experiments/smoke.yaml`**

```yaml
name: smoke
backend: local
backend_options:
  vram_gb: 0
models:
  - mock
inputs:
  - image: assets/examples/cube.png
params: {}
export: {}
seeds:
  - 42
```

- [ ] **Step 6: Crear la imagen de ejemplo que el config referencia**

```bash
mkdir -p assets/examples
python3 -c "
import struct, zlib, pathlib
w = h = 8
raw = b''.join(b'\x00' + bytes([40, 90, 160] * w) for _ in range(h))
def chunk(tag, data):
    body = tag + data
    return struct.pack('>I', len(data)) + body + struct.pack('>I', zlib.crc32(body))
png = (b'\x89PNG\r\n\x1a\n'
       + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
       + chunk(b'IDAT', zlib.compress(raw))
       + chunk(b'IEND', b''))
pathlib.Path('assets/examples/cube.png').write_bytes(png)
print('escrito', len(png), 'bytes')
"
```

- [ ] **Step 7: Correr los tests y el CLI de punta a punta**

Run:
```bash
.venv/bin/pytest tests/test_experiment.py -v
.venv/bin/python cli.py list
.venv/bin/python cli.py run experiments/smoke.yaml --runs-dir /tmp/og-runs
.venv/bin/python cli.py run experiments/smoke.yaml --runs-dir /tmp/og-runs
```
Expected: 4 tests PASS. El primer `run` imprime `[nuevo]`; el segundo, `[cache]`.

- [ ] **Step 8: Commit**

```bash
git add core/experiment.py cli.py experiments/ assets/ tests/test_experiment.py
git commit -m "feat(core): experimentos declarativos en YAML y CLI"
```

---

### Task 9: Integración continua sin GPU

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_layering.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: nada para el código; el test de capas protege la regla de acoplamiento del spec.

- [ ] **Step 1: Escribir el test de capas que falla**

Crear `tests/test_layering.py`:

```python
"""Protege las reglas de acoplamiento del spec.

`models/` no puede conocer la nube, y `backends/` no puede conocer los
modelos. Si esto se rompe, los dos ejes dejan de ser independientes.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLOUD_SDKS = {"boto3", "botocore", "runpod", "sagemaker"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", sorted((ROOT / "models").rglob("*.py")), ids=str)
def test_models_do_not_import_cloud_sdks(path):
    assert not (_imported_roots(path) & CLOUD_SDKS)


@pytest.mark.parametrize("path", sorted((ROOT / "backends").rglob("*.py")), ids=str)
def test_backends_do_not_import_models(path):
    assert "models" not in _imported_roots(path)


@pytest.mark.parametrize("path", sorted((ROOT / "core").rglob("*.py")), ids=str)
def test_core_depends_on_neither(path):
    roots = _imported_roots(path)
    assert "models" not in roots
    assert "backends" not in roots
```

- [ ] **Step 2: Correr el test y verificar que pasa**

Run: `.venv/bin/pytest tests/test_layering.py -v`
Expected: PASS. Este test se escribe para que pase desde el inicio: su valor es fallar el día que alguien viole la regla.

- [ ] **Step 3: Verificar que detecta una violación real**

```bash
echo "import boto3" >> models/mock.py
.venv/bin/pytest tests/test_layering.py -v
```
Expected: FAIL en `test_models_do_not_import_cloud_sdks`

Revertir:
```bash
git checkout models/mock.py
```

- [ ] **Step 4: Crear `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Instalar dependencias
        run: pip install -e ".[dev]"
      - name: Correr los tests
        run: pytest -v
      - name: Humo de punta a punta
        run: |
          python cli.py list
          python cli.py run experiments/smoke.yaml --runs-dir "$RUNNER_TEMP/runs"
          python cli.py run experiments/smoke.yaml --runs-dir "$RUNNER_TEMP/runs"
```

- [ ] **Step 5: Correr la suite completa**

Run: `.venv/bin/pytest -v`
Expected: PASS, sin fallos ni errores. Son unas 50 pruebas: 38 escritas a mano
más las que `test_layering.py` genera por parametrización, una por archivo de
`core/`, `models/` y `backends/`.

- [ ] **Step 6: Commit**

```bash
git add .github/ tests/test_layering.py
git commit -m "ci: suite sin GPU y test que protege las reglas de capas"
```

---

## Verificación final de la fase

Contra los criterios de aceptación del spec que aplican a esta fase:

| Criterio | Cómo se verifica |
|---|---|
| 1. Un config ejecuta modelo sobre backend sin código del par | `python cli.py run experiments/smoke.yaml` |
| 2. Agregar backend no toca `models/`, agregar modelo no toca `backends/` | `tests/test_layering.py` |
| 3. VRAM insuficiente falla antes de aprovisionar | `test_insufficient_vram_fails_before_submitting` |
| 4. Repetir un config usa caché | `test_rerunning_an_experiment_is_fully_cached` |
| 5. Toda corrida deja `provenance.json` | `test_execute_produces_a_run_directory` |
| 6. `teardown()` idempotente y siempre ejecutado | `test_teardown_is_idempotent`, `test_teardown_runs_even_when_generation_fails` |
| 7. Corre en CI sin GPU | `.github/workflows/ci.yml` |

El criterio 8 del spec —la conclusión sobre la hipótesis del renderizado— corresponde a la fase 3 y queda fuera de este plan.
