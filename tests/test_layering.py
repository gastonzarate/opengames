"""Protege las reglas de acoplamiento del spec.

`models/` no puede conocer la nube, y `backends/` no puede conocer los
modelos. Si esto se rompe, los dos ejes dejan de ser independientes.

## Alcance

Verifica **imports estáticos** capturados por análisis AST:
- `import boto3`
- `from boto3 import client`
- `from some.package import name` (cuando `some` es una raíz nube)

**No detecta** (limitaciones estructurales del AST):
- Imports dinámicos: `importlib.import_module("boto3")`, `__import__("boto3")`
- Imports relativos que suben más allá del nivel superior: `from ...pkg import x`
  (no son un vector real en la estructura actual de paquetes)

Las herramientas especializadas de análisis estático tienen las mismas
limitaciones. Para detectar imports dinámicos se requeriría análisis
de control-flow que está fuera del alcance.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLOUD_SDKS = {"boto3", "botocore", "runpod", "sagemaker"}

# Recolectar rutas ahora, no en tiempo de parametrización, para que un
# directorio vacío falle en import time en lugar de silenciosamente producir
# cero casos de test (que pytest marcaría como SKIPPED en lugar de FAILED).
_MODELS_PATHS = sorted((ROOT / "models").rglob("*.py"))
_BACKENDS_PATHS = sorted((ROOT / "backends").rglob("*.py"))
_CORE_PATHS = sorted((ROOT / "core").rglob("*.py"))

assert _MODELS_PATHS, "No se encontraron archivos .py en models/: el directorio puede faltar o estar vacío"
assert _BACKENDS_PATHS, "No se encontraron archivos .py en backends/: el directorio puede faltar o estar vacío"
assert _CORE_PATHS, "No se encontraron archivos .py en core/: el directorio puede faltar o estar vacío"


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", _MODELS_PATHS, ids=str)
def test_models_do_not_import_cloud_sdks(path):
    assert not (_imported_roots(path) & CLOUD_SDKS)


@pytest.mark.parametrize("path", _BACKENDS_PATHS, ids=str)
def test_backends_do_not_import_models(path):
    assert "models" not in _imported_roots(path)


@pytest.mark.parametrize("path", _CORE_PATHS, ids=str)
def test_core_depends_on_neither(path):
    roots = _imported_roots(path)
    assert "models" not in roots
    assert "backends" not in roots
