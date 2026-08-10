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
