# Fase 2 — TRELLIS.2 sobre SageMaker Async

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generar el primer asset 3D real del proyecto: una imagen de referencia entra, un GLB con materiales PBR sale, producido por TRELLIS.2 sobre un endpoint de SageMaker Asynchronous Inference.

**Architecture:** Una imagen Docker lleva el código y las seis extensiones CUDA compiladas; los pesos viven sin comprimir en S3 y SageMaker los baja a `/opt/ml/model`. Dentro del contenedor, una cáscara HTTP delgada expone `/ping` e `/invocations` sobre la misma función de generación que usaría cualquier otro backend. El backend `sagemaker` del harness traduce `submit`/`poll`/`fetch` a S3 más `InvokeEndpointAsync`.

**Tech Stack:** Docker, CUDA 12.4, PyTorch 2.6.0, Python 3.11, boto3, SageMaker Asynchronous Inference, ECR, S3.

## Global Constraints

- **Cuenta AWS `872154182820`, perfil `macacoai`, región `us-east-1`.** Verificar la identidad con `aws sts get-caller-identity` antes de cualquier operación que cree recursos.
- **La única cuota GPU disponible es `ml.g5.xlarge for endpoint usage = 1`.** Todo lo de EC2 está en cero. No intentar lanzar instancias EC2 ni usar otro tipo de instancia de SageMaker.
- **`ml.g5.xlarge` tiene 24 GB de VRAM, el mínimo exacto que declara TRELLIS.2.** Todas las corridas de esta fase usan `pipeline_type='512'` y `texture_size=2048`. No usar los valores del ejemplo oficial (`texture_size=4096`, `decimation_target=1000000`).
- **`models/` no puede importar SDKs de nube.** El adapter de TRELLIS.2 va en `models/`, así que no puede importar `boto3`. El transporte es responsabilidad de `backends/sagemaker.py`.
- **`core/` no puede importar de `models/` ni de `backends/`.** El test de capas de `tests/test_layering.py` lo verifica por AST y ya está en CI.
- **El endpoint se configura con `MinInstanceCount=0`** para que escale a cero y no facture cuando no hay trabajo.
- **Nunca escribir credenciales en código, configs ni reportes.** Las credenciales vienen del perfil de AWS; los identificadores de recursos que no son secretos (nombres de bucket, ARNs de rol) van en variables de entorno o en el config del experimento.
- Todo lo que no requiera GPU ni AWS tiene que correr en CI.

---

### Task 1: Imagen Docker con las seis extensiones CUDA

Es la tarea de mayor riesgo de toda la fase y la única que no depende de AWS. El README de TRELLIS.2 avisa que la instalación "puede tardar bastante" y sugiere instalar los flags de a uno ante fallas. Se valida en `gaston-pc`, que compila e importa bien con sus 8 GB aunque no pueda correr inferencia.

**Files:**
- Create: `docker/trellis2/Dockerfile`
- Create: `docker/trellis2/build.sh`
- Create: `docker/trellis2/smoke_imports.py`

**Interfaces:**
- Consumes: nada del harness.
- Produce: una imagen etiquetada `opengames/trellis2:<version>` que importa `trellis2`, `o_voxel`, `flash_attn`, `nvdiffrast`, `nvdiffrec`, `cumesh` y `flexgemm` sin error, y un `build.sh` que imprime el digest de la imagen construida.

- [ ] **Step 1: Escribir el script de humo de imports**

`docker/trellis2/smoke_imports.py`. Falla ruidosamente si falta cualquier extensión, y reporta todas las que faltan en vez de morir en la primera:

```python
"""Verifica que las seis extensiones CUDA y el paquete principal importen."""

import importlib
import sys

MODULOS = [
    "torch",
    "trellis2",
    "o_voxel",
    "flash_attn",
    "nvdiffrast",
    "nvdiffrec",
    "cumesh",
    "flexgemm",
]


def main() -> int:
    fallidos = []
    for nombre in MODULOS:
        try:
            importlib.import_module(nombre)
            print(f"  ok    {nombre}")
        except Exception as exc:
            fallidos.append((nombre, f"{type(exc).__name__}: {exc}"))
            print(f"  FALLA {nombre}: {type(exc).__name__}: {exc}")

    import torch

    print(f"\ntorch {torch.__version__} · CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"dispositivo: {torch.cuda.get_device_name(0)}")

    if fallidos:
        print(f"\n{len(fallidos)} módulo(s) no importan:")
        for nombre, error in fallidos:
            print(f"  - {nombre}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Escribir el Dockerfile**

`docker/trellis2/Dockerfile`. Base con CUDA 12.4 de desarrollo, porque `setup.sh` compila desde fuente y necesita `nvcc`:

```dockerfile
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" \
    PIP_NO_CACHE_DIR=1

# 8.6 = A10G (ml.g5) y RTX 3070 · 8.9 = L40S y RTX 4090 · 9.0 = H100.
# Compilar para las tres deja la imagen utilizable en gaston-pc y en AWS.

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build \
        python3.11 python3.11-dev python3-pip \
        libgl1 libglib2.0-0 libegl1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /opt
RUN git clone -b main --recursive https://github.com/microsoft/TRELLIS.2.git

WORKDIR /opt/TRELLIS.2
RUN python -m pip install --upgrade pip \
 && python -m pip install torch==2.6.0 torchvision --index-url https://download.pytorch.org/whl/cu124

# Las extensiones se instalan de a una y en capas separadas a propósito: si una
# falla, la caché de Docker conserva las anteriores y el ciclo de arreglo no
# vuelve a compilar todo desde cero.
RUN . ./setup.sh --basic
RUN . ./setup.sh --flash-attn
RUN . ./setup.sh --nvdiffrast
RUN . ./setup.sh --nvdiffrec
RUN . ./setup.sh --cumesh
RUN . ./setup.sh --o-voxel
RUN . ./setup.sh --flexgemm

COPY smoke_imports.py /opt/smoke_imports.py

ENV PYTHONPATH=/opt/TRELLIS.2
CMD ["python", "/opt/smoke_imports.py"]
```

- [ ] **Step 3: Escribir `build.sh`**

```bash
#!/usr/bin/env bash
# Construye la imagen e imprime su digest, que es lo que se registra como
# procedencia. Sin digest la corrida no es reproducible.
set -euo pipefail

VERSION="${1:-0.1.0}"
TAG="opengames/trellis2:${VERSION}"

docker build -t "$TAG" "$(dirname "$0")"
echo
echo "imagen : $TAG"
echo "digest : $(docker image inspect "$TAG" --format '{{index .Id}}')"
echo "tamaño : $(docker image inspect "$TAG" --format '{{.Size}}' | awk '{printf "%.1f GB\n", $1/1e9}')"
```

- [ ] **Step 4: Construir en gaston-pc**

La compilación es larga y la conexión es por SSH, así que conviene lanzarla en background y seguir el log. Recordá la maña del host: los comandos con comillas se rompen en `cmd` de Windows, así que va por script.

```bash
scp -r docker/trellis2 gaston-pc:C:/Windows/Temp/trellis2-build
ssh gaston-pc "wsl -d Ubuntu -e bash -lc 'cd /mnt/c/Windows/Temp/trellis2-build && bash build.sh 0.1.0 2>&1 | tee /tmp/build.log'"
```

Expected: la construcción termina sin error y `build.sh` imprime imagen, digest y tamaño.

Si una capa de `setup.sh` falla, el error queda acotado a esa extensión gracias a las capas separadas. Arreglar esa capa y reconstruir: las anteriores salen de caché.

- [ ] **Step 5: Correr el humo de imports con GPU**

```bash
ssh gaston-pc "wsl -d Ubuntu -e docker run --rm --gpus all opengames/trellis2:0.1.0"
```

Expected: las ocho líneas en `ok`, `CUDA disponible: True`, y el dispositivo reportado como `NVIDIA GeForce RTX 3070`.

Si `torch.cuda.is_available()` da `False`, el problema es el passthrough de GPU al contenedor en WSL2, no la imagen.

- [ ] **Step 6: Commit**

```bash
git add docker/
git commit -m "feat(docker): imagen de TRELLIS.2 con las seis extensiones CUDA"
```

---

### Task 2: Pesos de TRELLIS.2 en S3, sin comprimir

**Files:**
- Create: `scripts/subir_pesos.sh`

**Interfaces:**
- Consumes: nada del harness.
- Produce: los pesos bajo `s3://<bucket>/models/trellis2-4b/`, listos para declararse como `ModelDataSource`. El script imprime el prefijo S3 y la revisión de Hugging Face descargada, que es lo que se registra como procedencia.

Se suben **sin comprimir**: AWS lo [recomienda para modelos grandes](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-uncompressed.html) porque `ModelDataUrl` exige un `tar.gz` y paga la descompresión de 16 GB en cada arranque en frío.

- [ ] **Step 1: Escribir el script**

```bash
#!/usr/bin/env bash
# Descarga los pesos de TRELLIS.2 y los sube sin comprimir a S3.
# Idempotente: `aws s3 sync` no retransfiere lo que ya está.
set -euo pipefail

BUCKET="${OPENGAMES_BUCKET:?definí OPENGAMES_BUCKET}"
PROFILE="${AWS_PROFILE:-macacoai}"
REGION="${AWS_REGION:-us-east-1}"
LOCAL="${HOME}/.cache/opengames/trellis2-4b"
PREFIX="models/trellis2-4b"

# Para arrancar en 512 alcanzan estos checkpoints; los de 1024 y 1536 suman
# unos 5 GB que no se usan con pipeline_type='512'.
PATRONES=(
  "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"
  "ckpts/shape_enc_next_dc_f16c32_fp16.safetensors"
  "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors"
  "ckpts/tex_enc_next_dc_f16c32_fp16.safetensors"
  "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"
  "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors"
  "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors"
  "*.json"
)

mkdir -p "$LOCAL"
ARGS=()
for p in "${PATRONES[@]}"; do ARGS+=(--include "$p"); done

python -m pip install --quiet "huggingface_hub[cli]"
hf download microsoft/TRELLIS.2-4B --local-dir "$LOCAL" "${ARGS[@]}"

REVISION=$(git -C "$LOCAL" rev-parse HEAD 2>/dev/null || echo "desconocida")

aws s3 sync "$LOCAL" "s3://${BUCKET}/${PREFIX}/" \
    --profile "$PROFILE" --region "$REGION" --only-show-errors

echo
echo "prefijo S3 : s3://${BUCKET}/${PREFIX}/"
echo "revisión HF: ${REVISION}"
aws s3 ls "s3://${BUCKET}/${PREFIX}/ckpts/" --profile "$PROFILE" --region "$REGION" \
  --summarize --human-readable | tail -3
```

- [ ] **Step 2: Crear el bucket si no existe**

```bash
export AWS_PROFILE=macacoai AWS_PAGER=""
BUCKET="opengames-assets-872154182820"
aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null \
  || aws s3 mb "s3://$BUCKET" --region us-east-1
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

Expected: el bucket existe y tiene el acceso público bloqueado.

- [ ] **Step 3: Subir los pesos**

```bash
OPENGAMES_BUCKET=opengames-assets-872154182820 bash scripts/subir_pesos.sh
```

Expected: unos 11 GB bajo el prefijo, y el listado final mostrando los siete `.safetensors`.

- [ ] **Step 4: Commit**

```bash
git add scripts/subir_pesos.sh
git commit -m "feat(scripts): subida idempotente de los pesos de TRELLIS.2 a S3"
```

---

### Task 3: Adapter `models/trellis2.py`

**Files:**
- Create: `models/trellis2.py`
- Create: `tests/test_trellis2_adapter.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts` de `core.job`; `Modality`, `ModelSpec` de `core.model`; `register_model` de `core.registry`.
- Produce: clase `Trellis2Model` registrada como `"trellis2"`, con `describe()` declarando `min_vram_gb=24`, `accepts=[Modality.IMAGE]`, `produces=["glb", "preview"]`.

La importación de `trellis2` y `o_voxel` es **perezosa**, dentro de `load()`, no a nivel de módulo. Así el adapter se puede importar y consultar su `ModelSpec` en una máquina sin GPU ni pesos — que es lo que hacen el registro, el CLI y CI.

- [ ] **Step 1: Escribir el test que falla**

```python
import pytest

from core import registry
from core.model import Modality


@pytest.fixture(autouse=True)
def wiring():
    import importlib

    import models.trellis2

    registry.reset()
    importlib.reload(models.trellis2)
    yield
    registry.reset()


def test_describe_no_requiere_gpu_ni_pesos():
    """describe() tiene que funcionar en CI, sin CUDA ni checkpoints."""
    spec = registry.get_model("trellis2").describe()
    assert spec.min_vram_gb == 24
    assert spec.accepts == [Modality.IMAGE]
    assert "glb" in spec.produces
    assert spec.docker_image.startswith("opengames/trellis2:")


def test_el_modulo_no_importa_trellis2_al_cargarse():
    """La dependencia pesada es perezosa: import del adapter != import del modelo."""
    import ast
    import pathlib

    arbol = ast.parse(pathlib.Path("models/trellis2.py").read_text())
    nivel_superior = [
        n for n in arbol.body if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    raices = set()
    for nodo in nivel_superior:
        if isinstance(nodo, ast.Import):
            raices.update(a.name.split(".")[0] for a in nodo.names)
        elif nodo.module:
            raices.add(nodo.module.split(".")[0])
    assert "trellis2" not in raices
    assert "o_voxel" not in raices
    assert "torch" not in raices


def test_el_adapter_no_importa_sdks_de_nube():
    import pathlib

    fuente = pathlib.Path("models/trellis2.py").read_text()
    for sdk in ("boto3", "botocore", "sagemaker"):
        assert f"import {sdk}" not in fuente


def test_parametros_de_exportacion_por_defecto_caben_en_24gb():
    """24 GB es el mínimo exacto de la A10G: los defaults tienen que ser conservadores."""
    from models.trellis2 import EXPORTACION_POR_DEFECTO, GENERACION_POR_DEFECTO

    assert GENERACION_POR_DEFECTO["pipeline_type"] == "512"
    assert EXPORTACION_POR_DEFECTO["texture_size"] == 2048
    assert EXPORTACION_POR_DEFECTO["decimation_target"] <= 200_000
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_trellis2_adapter.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'models.trellis2'`

- [ ] **Step 3: Implementar el adapter**

```python
"""Adapter de TRELLIS.2: imagen → GLB con materiales PBR.

Las dependencias pesadas (`torch`, `trellis2`, `o_voxel`) se importan dentro de
`load()`, no a nivel de módulo. El registro, el CLI y CI importan este archivo
en máquinas sin GPU ni checkpoints, y solo necesitan `describe()`.
"""

import os
import time
from pathlib import Path
from typing import Any

from core.job import Artifacts, Job
from core.model import Modality, ModelSpec
from core.registry import register_model

VERSION_IMAGEN = "opengames/trellis2:0.1.0"
REVISION = "microsoft/TRELLIS.2-4B@main"

# 24 GB es el mínimo exacto de la ml.g5.xlarge (A10G). Los valores del ejemplo
# oficial —texture_size 4096 y decimation_target de un millón— no entran.
GENERACION_POR_DEFECTO: dict[str, Any] = {
    "pipeline_type": "512",
    "num_samples": 1,
}
EXPORTACION_POR_DEFECTO: dict[str, Any] = {
    "decimation_target": 50_000,
    "texture_size": 2048,
    "remesh": True,
    "remesh_band": 1,
    "remesh_project": 0,
}

# Ruta donde SageMaker deja los artefactos declarados con ModelDataSource.
RAIZ_PESOS = Path(os.environ.get("OPENGAMES_PESOS", "/opt/ml/model"))


@register_model("trellis2")
class Trellis2Model:
    def __init__(self) -> None:
        self._pipeline = None

    def describe(self) -> ModelSpec:
        return ModelSpec(
            name="trellis2",
            revision=REVISION,
            min_vram_gb=24,
            accepts=[Modality.IMAGE],
            produces=["glb", "preview"],
            docker_image=VERSION_IMAGEN,
        )

    def load(self) -> None:
        if self._pipeline is not None:
            return
        import torch  # noqa: F401  fuerza la inicialización de CUDA
        from trellis2.pipelines import Trellis2ImageTo3DPipeline

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

        self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(RAIZ_PESOS))
        self._pipeline.cuda()

    def generate(self, job: Job, workdir: Path) -> Artifacts:
        import o_voxel
        from PIL import Image

        self.load()
        inicio = time.perf_counter()
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        generacion = {**GENERACION_POR_DEFECTO, **job.params, "seed": job.seed}
        exportacion = {**EXPORTACION_POR_DEFECTO, **job.export}

        imagen = Image.open(job.inputs["image"])
        malla = self._pipeline.run(imagen, **generacion)[0]
        malla.simplify(16_777_216)  # límite de nvdiffrast

        glb = o_voxel.postprocess.to_glb(
            vertices=malla.vertices,
            faces=malla.faces,
            attr_volume=malla.attrs,
            coords=malla.coords,
            attr_layout=malla.layout,
            voxel_size=malla.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            verbose=False,
            **exportacion,
        )
        destino = workdir / "sample.glb"
        glb.export(str(destino), extension_webp=True)

        return Artifacts(
            files={"glb": destino},
            metrics={
                "duration_s": time.perf_counter() - inicio,
                "peak_vram_gb": _pico_vram_gb(),
            },
        )


def _pico_vram_gb() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        return 0.0
```

- [ ] **Step 4: Correr los tests y verificar que pasan**

Run: `.venv/bin/pytest tests/test_trellis2_adapter.py tests/test_layering.py -v`
Expected: PASS. El test de capas tiene que seguir en verde: `models/trellis2.py` es un archivo nuevo dentro de `models/` y ahora también lo cubre.

- [ ] **Step 5: Commit**

```bash
git add models/trellis2.py tests/test_trellis2_adapter.py
git commit -m "feat(models): adapter de TRELLIS.2 con importación perezosa"
```

---

### Task 4: Cáscara HTTP para el contrato de SageMaker

**Files:**
- Create: `docker/trellis2/servidor.py`
- Modify: `docker/trellis2/Dockerfile`
- Create: `tests/test_servidor_sagemaker.py`

**Interfaces:**
- Consumes: el adapter `Trellis2Model` de `models.trellis2`.
- Produce: un servidor que responde `GET /ping` con 200 y `POST /invocations` con el GLB, escuchando en el 8080. Función `crear_app(adapter)` para poder testearla con un adapter simulado.

SageMaker exige exactamente ese contrato. La cáscara es delgada a propósito: toda la lógica de generación vive en el adapter, así que la misma imagen sirve para EC2 o RunPod cambiando solo el entrypoint.

- [ ] **Step 1: Escribir el test que falla**

Se testea con el adapter simulado, sin GPU, así que corre en CI:

```python
import base64
import json
from pathlib import Path

import pytest

from core.job import Artifacts, Job
from core.model import Modality, ModelSpec


class AdapterFalso:
    def __init__(self):
        self.llamadas = []

    def describe(self):
        return ModelSpec(name="falso", revision="0", min_vram_gb=0,
                         accepts=[Modality.IMAGE], produces=["glb"],
                         docker_image="falso:0")

    def load(self):
        pass

    def generate(self, job: Job, workdir: Path) -> Artifacts:
        self.llamadas.append(job)
        destino = Path(workdir) / "sample.glb"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(b"glTF-falso")
        return Artifacts(files={"glb": destino}, metrics={"duration_s": 0.1})


@pytest.fixture
def cliente():
    import sys

    sys.path.insert(0, "docker/trellis2")
    from servidor import crear_app

    adapter = AdapterFalso()
    app = crear_app(adapter)
    app.config["TESTING"] = True
    return app.test_client(), adapter


def test_ping_responde_200(cliente):
    c, _ = cliente
    assert c.get("/ping").status_code == 200


def test_invocations_devuelve_el_glb(cliente):
    c, adapter = cliente
    payload = {"image_b64": base64.b64encode(b"png-falso").decode(),
               "params": {"pipeline_type": "512"}, "seed": 7}
    r = c.post("/invocations", data=json.dumps(payload),
               content_type="application/json")
    assert r.status_code == 200
    cuerpo = json.loads(r.data)
    assert base64.b64decode(cuerpo["glb_b64"]) == b"glTF-falso"
    assert cuerpo["metrics"]["duration_s"] == 0.1
    assert adapter.llamadas[0].seed == 7
    assert adapter.llamadas[0].params["pipeline_type"] == "512"


def test_invocations_sin_imagen_da_400(cliente):
    c, _ = cliente
    r = c.post("/invocations", data=json.dumps({}), content_type="application/json")
    assert r.status_code == 400
    assert "image_b64" in json.loads(r.data)["error"]


def test_un_fallo_de_generacion_da_500_con_mensaje(cliente):
    c, adapter = cliente

    def explota(job, workdir):
        raise RuntimeError("sin memoria")

    adapter.generate = explota
    payload = {"image_b64": base64.b64encode(b"x").decode()}
    r = c.post("/invocations", data=json.dumps(payload), content_type="application/json")
    assert r.status_code == 500
    assert "sin memoria" in json.loads(r.data)["error"]
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_servidor_sagemaker.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'servidor'`

- [ ] **Step 3: Implementar el servidor**

```python
"""Contrato de SageMaker: GET /ping y POST /invocations en el 8080.

Cáscara delgada sobre el adapter. La lógica de generación no vive acá, para que
la misma imagen sirva en otros backends cambiando solo el entrypoint.
"""

import base64
import json
import tempfile
from pathlib import Path

from flask import Flask, Response, request

from core.job import Job


def crear_app(adapter) -> Flask:
    app = Flask(__name__)

    @app.route("/ping", methods=["GET"])
    def ping() -> Response:
        # SageMaker considera vivo al contenedor con un 200. No se carga el
        # modelo acá: la carga tarda minutos y el health check tiene su propio
        # tiempo de espera.
        return Response(status=200)

    @app.route("/invocations", methods=["POST"])
    def invocations() -> Response:
        try:
            cuerpo = request.get_json(force=True, silent=True) or {}
        except Exception:
            cuerpo = {}

        if "image_b64" not in cuerpo:
            return _json({"error": "falta 'image_b64' en el cuerpo del pedido"}, 400)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            entrada = tmp / "entrada.png"
            entrada.write_bytes(base64.b64decode(cuerpo["image_b64"]))

            job = Job(
                model="trellis2",
                inputs={"image": entrada},
                params=cuerpo.get("params", {}),
                export=cuerpo.get("export", {}),
                seed=cuerpo.get("seed", 42),
            )
            try:
                artefactos = adapter.generate(job, tmp / "salida")
            except Exception as exc:
                return _json({"error": f"{type(exc).__name__}: {exc}"}, 500)

            glb = Path(artefactos.files["glb"]).read_bytes()

        return _json(
            {
                "glb_b64": base64.b64encode(glb).decode(),
                "metrics": artefactos.metrics,
            },
            200,
        )

    return app


def _json(cuerpo: dict, codigo: int) -> Response:
    return Response(json.dumps(cuerpo), status=codigo, mimetype="application/json")


if __name__ == "__main__":
    from models.trellis2 import Trellis2Model

    crear_app(Trellis2Model()).run(host="0.0.0.0", port=8080)
```

- [ ] **Step 4: Agregar el servidor y el harness a la imagen**

En `docker/trellis2/Dockerfile`, reemplazar el bloque final (`COPY smoke_imports.py` y el `CMD`) por:

```dockerfile
RUN python -m pip install flask gunicorn pydantic PyYAML

COPY smoke_imports.py /opt/smoke_imports.py
COPY servidor.py /opt/app/servidor.py
COPY core /opt/app/core
COPY models /opt/app/models

ENV PYTHONPATH=/opt/TRELLIS.2:/opt/app
WORKDIR /opt/app

# Un solo worker y sin timeout: la generación tarda minutos y el modelo ocupa
# casi toda la VRAM de la A10G, así que dos workers no entran.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "0", \
     "servidor:crear_app_produccion()"]
```

Y agregar al final de `servidor.py`:

```python
def crear_app_produccion() -> Flask:
    """Punto de entrada de gunicorn."""
    from models.trellis2 import Trellis2Model

    return crear_app(Trellis2Model())
```

El `build.sh` tiene que copiar `core/` y `models/` al contexto de build. Agregar antes del `docker build`:

```bash
RAIZ="$(cd "$(dirname "$0")/../.." && pwd)"
rsync -a --delete "$RAIZ/core" "$RAIZ/models" "$(dirname "$0")/"
```

y añadir a `.gitignore`:

```
docker/trellis2/core/
docker/trellis2/models/
```

- [ ] **Step 5: Declarar `flask` como dependencia de desarrollo**

Sin esto el CI falla: el workflow instala con `pip install -e ".[dev]"` y no tendría
flask para importar el servidor. En `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "flask>=3.0"]
```

Luego `.venv/bin/pip install -e ".[dev]"`.

`boto3` **no** hace falta como dependencia: el backend de la Task 5 lo importa de forma
perezosa y sus tests usan clientes falsos que no lo necesitan. En producción viene dentro
de la imagen.

- [ ] **Step 6: Correr los tests y reconstruir**

```bash
.venv/bin/pytest tests/test_servidor_sagemaker.py -v
.venv/bin/pytest -q
bash docker/trellis2/build.sh 0.2.0
```
Expected: 4 tests nuevos en verde, la suite completa sin regresiones, y la imagen `0.2.0`
construida.

- [ ] **Step 7: Commit**

```bash
git add docker/ tests/test_servidor_sagemaker.py .gitignore pyproject.toml
git commit -m "feat(docker): cáscara HTTP con el contrato /ping e /invocations de SageMaker"
```

---

### Task 5: Backend `backends/sagemaker.py`

**Files:**
- Create: `backends/sagemaker.py`
- Create: `tests/test_sagemaker_backend.py`

**Interfaces:**
- Consumes: `Job`, `Artifacts`, `RunHandle`, `RunStatus` de `core.job`; `BackendSpec` de `core.backend`; `register_backend` de `core.registry`.
- Produce: clase `SageMakerBackend` registrada como `"sagemaker"`, con constructor `SageMakerBackend(endpoint_name, bucket, prefix="async", vram_gb=24, region="us-east-1", cliente_sm=None, cliente_s3=None)`.

Los dos clientes se inyectan por constructor para poder testear sin AWS. En producción se construyen con boto3 si vienen en `None`.

- [ ] **Step 1: Escribir el test que falla**

```python
import base64
import json
from pathlib import Path

import pytest

from core.job import Job, RunStatus


class S3Falso:
    def __init__(self):
        self.objetos = {}

    def upload_file(self, ruta, bucket, key):
        self.objetos[f"{bucket}/{key}"] = Path(ruta).read_bytes()

    def put_object(self, Bucket, Key, Body):
        self.objetos[f"{Bucket}/{Key}"] = Body

    def head_object(self, Bucket, Key):
        # Se lanza una excepción común y no un ClientError de botocore a
        # propósito: `_existe()` atrapa `Exception`, y así el test no necesita
        # boto3 instalado para correr en CI.
        clave = f"{Bucket}/{Key}"
        if clave not in self.objetos:
            raise KeyError(f"404 {clave}")
        return {"ContentLength": len(self.objetos[clave])}

    def delete_object(self, Bucket, Key):
        self.objetos.pop(f"{Bucket}/{Key}", None)

    def get_object(self, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.objetos[f"{Bucket}/{Key}"])}


class SMFalso:
    def __init__(self, s3, bucket, responder=True):
        self.s3, self.bucket, self.responder = s3, bucket, responder
        self.invocaciones = []

    def invoke_endpoint_async(self, EndpointName, InputLocation, ContentType, InferenceId):
        self.invocaciones.append(InferenceId)
        salida = f"{self.bucket}/async/salida/{InferenceId}.out"
        if self.responder:
            self.s3.objetos[salida] = json.dumps(
                {"glb_b64": base64.b64encode(b"glTF-real").decode(),
                 "metrics": {"duration_s": 42.0}}
            ).encode()
        return {"OutputLocation": f"s3://{salida}", "InferenceId": InferenceId}


@pytest.fixture
def backend_falso(tmp_path):
    from backends.sagemaker import SageMakerBackend

    s3 = S3Falso()
    sm = SMFalso(s3, "bucket-test")
    b = SageMakerBackend(endpoint_name="ep", bucket="bucket-test",
                         cliente_sm=sm, cliente_s3=s3)
    img = tmp_path / "ref.png"
    img.write_bytes(b"png")
    return b, s3, sm, img


def test_capabilities_declara_la_vram_de_la_a10g(backend_falso):
    b, *_ = backend_falso
    caps = b.capabilities()
    assert caps.vram_gb == 24
    assert caps.ephemeral is True


def test_submit_sube_la_entrada_e_invoca(backend_falso):
    b, s3, sm, img = backend_falso
    h = b.submit(Job(model="trellis2", inputs={"image": img}))
    assert len(sm.invocaciones) == 1
    assert any(k.endswith(".json") for k in s3.objetos)
    assert h.remote_id == sm.invocaciones[0]


def test_ciclo_completo_devuelve_el_glb(backend_falso, tmp_path):
    b, s3, sm, img = backend_falso
    h = b.submit(Job(model="trellis2", inputs={"image": img}))
    assert b.poll(h) is RunStatus.SUCCEEDED
    art = b.fetch(h, tmp_path / "out")
    assert art.files["glb"].read_bytes() == b"glTF-real"
    assert art.metrics["duration_s"] == 42.0


def test_poll_devuelve_running_mientras_no_haya_salida(backend_falso):
    from backends.sagemaker import SageMakerBackend

    s3 = S3Falso()
    sm = SMFalso(s3, "bucket-test", responder=False)
    b = SageMakerBackend(endpoint_name="ep", bucket="bucket-test",
                         cliente_sm=sm, cliente_s3=s3)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        f.write(b"png"); f.flush()
        h = b.submit(Job(model="trellis2", inputs={"image": Path(f.name)}))
    assert b.poll(h) is RunStatus.RUNNING


def test_un_objeto_de_error_marca_failed(backend_falso):
    b, s3, sm, img = backend_falso
    h = b.submit(Job(model="trellis2", inputs={"image": img}))
    del s3.objetos[f"bucket-test/async/salida/{h.remote_id}.out"]
    s3.objetos[f"bucket-test/async/error/{h.remote_id}.out"] = b'{"error":"sin memoria"}'
    assert b.poll(h) is RunStatus.FAILED
    assert "sin memoria" in b.error(h)


def test_teardown_es_idempotente_y_no_borra_el_endpoint(backend_falso):
    b, s3, sm, img = backend_falso
    h = b.submit(Job(model="trellis2", inputs={"image": img}))
    b.teardown(h)
    b.teardown(h)
    assert sm.invocaciones  # el endpoint sigue existiendo, no se toca
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `.venv/bin/pytest tests/test_sagemaker_backend.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'backends.sagemaker'`

- [ ] **Step 3: Implementar el backend**

```python
"""Backend de SageMaker Asynchronous Inference.

El recurso caro es el endpoint, que vive entre corridas: recrearlo por corrida
tardaría minutos. Por eso `teardown()` solo limpia los objetos temporales de S3
y el ciclo de vida del endpoint es una operación aparte, en `scripts/endpoint.py`.
Es una asimetría deliberada respecto de los backends efímeros.
"""

import base64
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from core.backend import BackendSpec
from core.job import Artifacts, Job, RunHandle, RunStatus
from core.registry import register_backend


@dataclass
class _Corrida:
    inference_id: str
    key_entrada: str
    error: str = ""
    metrics: dict = field(default_factory=dict)


@register_backend("sagemaker")
class SageMakerBackend:
    def __init__(
        self,
        endpoint_name: str,
        bucket: str,
        prefix: str = "async",
        vram_gb: int = 24,
        region: str = "us-east-1",
        cliente_sm=None,
        cliente_s3=None,
    ) -> None:
        self.endpoint_name = endpoint_name
        self.bucket = bucket
        self.prefix = prefix
        self.vram_gb = vram_gb
        self._sm = cliente_sm
        self._s3 = cliente_s3
        self._region = region
        self._corridas: dict[str, _Corrida] = {}

    def _sm_cliente(self):
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker-runtime", region_name=self._region)
        return self._sm

    def _s3_cliente(self):
        if self._s3 is None:
            import boto3

            self._s3 = boto3.client("s3", region_name=self._region)
        return self._s3

    def capabilities(self) -> BackendSpec:
        return BackendSpec(name="sagemaker", vram_gb=self.vram_gb, ephemeral=True)

    def submit(self, job: Job) -> RunHandle:
        inference_id = uuid.uuid4().hex
        key = f"{self.prefix}/entrada/{inference_id}.json"

        payload = {
            "image_b64": base64.b64encode(Path(job.inputs["image"]).read_bytes()).decode(),
            "params": job.params,
            "export": job.export,
            "seed": job.seed,
        }
        self._s3_cliente().put_object(
            Bucket=self.bucket, Key=key, Body=json.dumps(payload).encode()
        )
        self._sm_cliente().invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{self.bucket}/{key}",
            ContentType="application/json",
            InferenceId=inference_id,
        )
        self._corridas[inference_id] = _Corrida(inference_id, key)
        return RunHandle(backend="sagemaker", run_id=inference_id, remote_id=inference_id)

    def _existe(self, key: str) -> bool:
        try:
            self._s3_cliente().head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def poll(self, handle: RunHandle) -> RunStatus:
        iid = handle.remote_id
        if self._existe(f"{self.prefix}/salida/{iid}.out"):
            return RunStatus.SUCCEEDED
        key_error = f"{self.prefix}/error/{iid}.out"
        if self._existe(key_error):
            cuerpo = self._s3_cliente().get_object(Bucket=self.bucket, Key=key_error)["Body"].read()
            self._corridas[iid].error = cuerpo.decode(errors="replace")
            return RunStatus.FAILED
        return RunStatus.RUNNING

    def error(self, handle: RunHandle) -> str:
        corrida = self._corridas.get(handle.remote_id)
        return corrida.error if corrida else ""

    def fetch(self, handle: RunHandle, dest: Path) -> Artifacts:
        iid = handle.remote_id
        cuerpo = self._s3_cliente().get_object(
            Bucket=self.bucket, Key=f"{self.prefix}/salida/{iid}.out"
        )["Body"].read()
        respuesta = json.loads(cuerpo)

        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        destino = dest / "sample.glb"
        destino.write_bytes(base64.b64decode(respuesta["glb_b64"]))

        return Artifacts(files={"glb": destino}, metrics=respuesta.get("metrics", {}))

    def teardown(self, handle: RunHandle) -> None:
        """Borra los objetos temporales. NO toca el endpoint: vive entre corridas."""
        corrida = self._corridas.get(handle.remote_id)
        if corrida is None:
            return
        for key in (
            corrida.key_entrada,
            f"{self.prefix}/salida/{corrida.inference_id}.out",
            f"{self.prefix}/error/{corrida.inference_id}.out",
        ):
            try:
                self._s3_cliente().delete_object(Bucket=self.bucket, Key=key)
            except Exception:
                pass
```

Agregar `delete_object` al `S3Falso` del test:

```python
    def delete_object(self, Bucket, Key):
        self.objetos.pop(f"{Bucket}/{Key}", None)
```

- [ ] **Step 4: Correr los tests**

Run: `.venv/bin/pytest tests/test_sagemaker_backend.py tests/test_layering.py -v`
Expected: 7 tests nuevos en verde. El test de capas también: `backends/sagemaker.py` importa boto3, que está permitido en `backends/`, y no importa de `models/`.

- [ ] **Step 5: Commit**

```bash
git add backends/sagemaker.py tests/test_sagemaker_backend.py
git commit -m "feat(backends): backend de SageMaker Asynchronous Inference"
```

---

### Task 6: Ciclo de vida del endpoint

**Files:**
- Create: `scripts/endpoint.py`

**Interfaces:**
- Consumes: boto3.
- Produce: un CLI con subcomandos `crear`, `estado` y `borrar`, que gestiona el modelo, la configuración y el endpoint de SageMaker.

Es una operación explícita y separada del harness, por lo dicho en la Task 5: el endpoint vive entre corridas.

- [ ] **Step 1: Escribir el script**

```python
"""Ciclo de vida del endpoint de SageMaker Async.

    python scripts/endpoint.py crear --imagen <uri-ecr> --pesos s3://... --rol <arn>
    python scripts/endpoint.py estado
    python scripts/endpoint.py borrar

`MinInstanceCount=0` es lo que evita pagar GPU ociosa: sin trabajo el endpoint
escala a cero. El precio es un arranque en frío de varios minutos mientras
SageMaker baja los pesos a /opt/ml/model.
"""

import argparse
import sys

import boto3

NOMBRE = "opengames-trellis2"
INSTANCIA = "ml.g5.xlarge"  # única con cuota disponible: A10G, 24 GB


def crear(sm, args) -> int:
    sm.create_model(
        ModelName=NOMBRE,
        ExecutionRoleArn=args.rol,
        PrimaryContainer={
            "Image": args.imagen,
            "ModelDataSource": {
                "S3DataSource": {
                    "S3Uri": args.pesos,
                    "S3DataType": "S3Prefix",
                    "CompressionType": "None",
                }
            },
            "Environment": {"OPENGAMES_PESOS": "/opt/ml/model"},
        },
    )
    sm.create_endpoint_config(
        EndpointConfigName=NOMBRE,
        ProductionVariants=[
            {
                "VariantName": "principal",
                "ModelName": NOMBRE,
                "InstanceType": INSTANCIA,
                "InitialInstanceCount": 1,
                "ModelDataDownloadTimeoutInSeconds": 1800,
                "ContainerStartupHealthCheckTimeoutInSeconds": 1800,
            }
        ],
        AsyncInferenceConfig={
            "OutputConfig": {
                "S3OutputPath": f"s3://{args.bucket}/async/salida/",
                "S3FailurePath": f"s3://{args.bucket}/async/error/",
            },
            "ClientConfig": {"MaxConcurrentInvocationsPerInstance": 1},
        },
    )
    sm.create_endpoint(EndpointName=NOMBRE, EndpointConfigName=NOMBRE)
    print(f"endpoint {NOMBRE} creándose sobre {INSTANCIA}")
    print("seguí el estado con: python scripts/endpoint.py estado")
    return 0


def estado(sm, _) -> int:
    d = sm.describe_endpoint(EndpointName=NOMBRE)
    print(f"estado: {d['EndpointStatus']}")
    if d.get("FailureReason"):
        print(f"motivo: {d['FailureReason']}")
    return 0


def borrar(sm, _) -> int:
    for fn, nombre in (
        (sm.delete_endpoint, "endpoint"),
        (sm.delete_endpoint_config, "config"),
        (sm.delete_model, "modelo"),
    ):
        try:
            fn(**{"EndpointName" if nombre == "endpoint" else
                  "EndpointConfigName" if nombre == "config" else "ModelName": NOMBRE})
            print(f"  {nombre} borrado")
        except Exception as exc:
            print(f"  {nombre}: {type(exc).__name__}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="endpoint")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("crear")
    c.add_argument("--imagen", required=True, help="URI de la imagen en ECR")
    c.add_argument("--pesos", required=True, help="prefijo S3 de los pesos")
    c.add_argument("--rol", required=True, help="ARN del rol de ejecución")
    c.add_argument("--bucket", required=True)
    sub.add_parser("estado")
    sub.add_parser("borrar")

    args = p.parse_args()
    sm = boto3.client("sagemaker", region_name="us-east-1")
    return {"crear": crear, "estado": estado, "borrar": borrar}[args.cmd](sm, args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Crear el rol de ejecución**

```bash
export AWS_PROFILE=macacoai AWS_PAGER=""
cat > /tmp/confianza.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF
aws iam create-role --role-name OpenGamesSageMakerExec \
  --assume-role-policy-document file:///tmp/confianza.json
aws iam attach-role-policy --role-name OpenGamesSageMakerExec \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam attach-role-policy --role-name OpenGamesSageMakerExec \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam get-role --role-name OpenGamesSageMakerExec --query 'Role.Arn' --output text
```

Expected: el ARN del rol. Guardarlo, lo necesita el paso de creación.

- [ ] **Step 3: Publicar la imagen en ECR**

```bash
export AWS_PROFILE=macacoai AWS_PAGER=""
CUENTA=872154182820; REGION=us-east-1; REPO=opengames/trellis2
aws ecr describe-repositories --repository-names "$REPO" --region $REGION 2>/dev/null \
  || aws ecr create-repository --repository-name "$REPO" --region $REGION
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin "$CUENTA.dkr.ecr.$REGION.amazonaws.com"
docker tag opengames/trellis2:0.2.0 "$CUENTA.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0"
docker push "$CUENTA.dkr.ecr.$REGION.amazonaws.com/$REPO:0.2.0"
aws ecr describe-images --repository-name "$REPO" --image-ids imageTag=0.2.0 \
  --region $REGION --query 'imageDetails[0].imageDigest' --output text
```

Expected: el digest de la imagen en ECR. **Ese digest es el que va a `provenance.json`**, no el tag: es la deuda registrada en `docs/superpowers/deuda-tecnica-fase1.md` y esta fase es donde hay que cerrarla.

El push tiene que hacerse desde `gaston-pc`, que es donde está construida la imagen.

- [ ] **Step 4: Crear el endpoint**

```bash
python scripts/endpoint.py crear \
  --imagen 872154182820.dkr.ecr.us-east-1.amazonaws.com/opengames/trellis2:0.2.0 \
  --pesos s3://opengames-assets-872154182820/models/trellis2-4b/ \
  --rol <arn-del-paso-2> \
  --bucket opengames-assets-872154182820
```

Luego consultar hasta que salga `InService`:

```bash
python scripts/endpoint.py estado
```

Expected: pasa de `Creating` a `InService`. Puede tardar 15 minutos o más: SageMaker baja 11 GB de pesos y arranca el contenedor. Si queda en `Failed`, el motivo aparece en la salida y los logs están en CloudWatch, en el grupo `/aws/sagemaker/Endpoints/opengames-trellis2`.

- [ ] **Step 5: Commit**

```bash
git add scripts/endpoint.py
git commit -m "feat(scripts): ciclo de vida del endpoint de SageMaker Async"
```

---

### Task 7: Primera generación real

**Files:**
- Create: `experiments/trellis2-primera.yaml`
- Modify: `cli.py`
- Modify: `docs/superpowers/deuda-tecnica-fase1.md`

**Interfaces:**
- Consumes: todo lo anterior.
- Produce: un GLB real en `runs/<run_id>/outputs/`, generado por TRELLIS.2 a partir de una imagen de referencia.

- [ ] **Step 1: Registrar el modelo y el backend nuevos en el CLI**

En `cli.py`, junto a los imports existentes:

```python
import backends.sagemaker  # noqa: F401  registra el backend
import models.trellis2  # noqa: F401  registra el modelo
```

- [ ] **Step 2: Escribir el config del experimento**

`experiments/trellis2-primera.yaml`:

```yaml
name: trellis2-primera
backend: sagemaker
backend_options:
  endpoint_name: opengames-trellis2
  bucket: opengames-assets-872154182820
  vram_gb: 24
models:
  - trellis2
inputs:
  - image: ../assets/examples/referencia.png
params:
  pipeline_type: "512"
export:
  texture_size: 2048
  decimation_target: 50000
seeds:
  - 42
```

- [ ] **Step 3: Conseguir una imagen de referencia**

Tiene que ser un objeto centrado con fondo limpio. `rembg` corre dentro del pipeline y quita el fondo, así que una foto normal sirve. Guardar como `assets/examples/referencia.png`.

- [ ] **Step 4: Correr**

```bash
export AWS_PROFILE=macacoai
.venv/bin/python cli.py run experiments/trellis2-primera.yaml --runs-dir runs
```

Expected: imprime `[nuevo] <run_id>`. La primera invocación paga el arranque en frío, así que puede tardar varios minutos.

Verificar el resultado:

```bash
.venv/bin/python - <<'EOF'
import json, struct, pathlib, sys
glb = sorted(pathlib.Path("runs").rglob("sample.glb"))[-1]
raw = glb.read_bytes()
magic, ver, total = struct.unpack("<III", raw[:12])
assert magic == 0x46546C67 and total == len(raw), "GLB inválido"
off = 12
while off < len(raw):
    ln, kind = struct.unpack("<II", raw[off:off+8])
    if kind == 0x4E4F534A:
        g = json.loads(raw[off+8:off+8+ln].decode("utf8")); break
    off += 8 + ln
tris = sum(g["accessors"][p["indices"]]["count"] // 3
           for m in g["meshes"] for p in m["primitives"] if "indices" in p)
mat = g["materials"][0]["pbrMetallicRoughness"]
print(f"{glb}  {len(raw)/1e6:.1f} MB")
print(f"triángulos: {tris:,}")
print(f"mapas PBR : {[k for k in mat if k.endswith('Texture')]}")
EOF
```

Expected: un GLB válido, con un recuento de triángulos cercano al `decimation_target` de 50.000, y **`baseColorTexture` junto a `metallicRoughnessTexture`** — que son los dos mapas separados que motivaron elegir TRELLIS.2 sobre TRELLIS v1.

- [ ] **Step 5: Correr una segunda vez y confirmar el caché**

```bash
.venv/bin/python cli.py run experiments/trellis2-primera.yaml --runs-dir runs
```
Expected: `[cache] <mismo run_id>`, sin invocar el endpoint. Es la primera vez que el caché evita gasto real de GPU.

- [ ] **Step 6: Registrar el digest y cerrar la deuda**

Verificar que `runs/<run_id>/provenance.json` tenga el digest de ECR y no el tag. Si tiene el tag, actualizar `VERSION_IMAGEN` en `models/trellis2.py` para que sea la referencia por digest (`...@sha256:...`) y anotar en `docs/superpowers/deuda-tecnica-fase1.md` que el ítem quedó cerrado.

- [ ] **Step 7: Apagar el endpoint**

`MinInstanceCount=0` evita el costo de GPU ociosa, pero mientras el endpoint existe hay costo asociado. Si no se va a usar por un rato:

```bash
python scripts/endpoint.py borrar
```

- [ ] **Step 8: Commit**

```bash
git add experiments/ cli.py assets/ docs/ models/
git commit -m "feat: primera generación real de un asset con TRELLIS.2 sobre SageMaker"
```

---

## Verificación final de la fase

| Qué | Cómo |
|---|---|
| Las seis extensiones CUDA compilan e importan | Task 1, paso 5, sobre GPU real |
| El adapter se puede importar sin GPU ni pesos | `tests/test_trellis2_adapter.py`, en CI |
| El contrato de SageMaker se cumple | `tests/test_servidor_sagemaker.py`, en CI |
| El backend traduce bien a S3 y `InvokeEndpointAsync` | `tests/test_sagemaker_backend.py`, en CI |
| Las reglas de capas siguen valiendo con los archivos nuevos | `tests/test_layering.py`, en CI |
| Sale un GLB real con mapas PBR separados | Task 7, paso 4 |
| El caché evita una invocación paga | Task 7, paso 5 |
| La procedencia registra el digest, no el tag | Task 7, paso 6 |

## Riesgos

| Riesgo | Mitigación |
|---|---|
| Alguna extensión CUDA no compila | Capas separadas por extensión en el Dockerfile: el error queda acotado y la caché conserva el resto. El README sugiere instalar de a un flag |
| 24 GB no alcanzan ni siquiera a 512³ | Bajar `texture_size` a 1024 y `decimation_target`. Si aun así no entra, TRELLIS.2 queda fuera de la cuota disponible y hay que pedir la de EC2 |
| El endpoint queda en `Failed` | Los logs están en CloudWatch, grupo `/aws/sagemaker/Endpoints/opengames-trellis2`. Las causas más comunes son el timeout de descarga de pesos y que `/ping` no responda a tiempo — por eso ambos timeouts están en 1800 s |
| El arranque en frío excede el timeout de invocación | `InvocationTimeoutSeconds` admite hasta 3600 s. Si no alcanza, mantener una instancia mínima encendida durante la sesión de trabajo y apagarla al terminar |
| Costo inesperado | El endpoint es el único recurso que factura. Task 7 paso 7 lo borra. Conviene revisar Cost Explorer al día siguiente de la primera corrida |
