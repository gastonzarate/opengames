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


def test_exists_rejects_truncated_metrics(tmp_path, spec, image):
    """metrics.json truncado o corrupto no cuenta como caché válida."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Escribir un metrics.json válido
    store.create(run_id)
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    # Truncar el archivo (simular interrupción a mitad de escritura)
    metrics_path = store._dir(run_id) / "metrics.json"
    metrics_path.write_text("{")  # JSON inválido

    # Debe retornar False porque el JSON es corrupto
    assert not store.exists(run_id)


def test_load_artifacts_on_nonexistent_run_returns_empty(tmp_path, spec, image):
    """load_artifacts sobre un run_id inexistente retorna Artifacts vacío sin crear dirs."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Directorio no existe
    assert not (store._dir(run_id)).exists()

    # Cargar artifacts no debe crear directorios
    artifacts = store.load_artifacts(run_id)

    # El directorio raíz no debe haber sido creado
    assert not (store._dir(run_id)).exists()
    assert artifacts.metrics == {}
    assert artifacts.files == {}


def test_retry_cleans_old_outputs(tmp_path, spec, image):
    """Reintento sobre run_id con corrida exitosa anterior no arrastra outputs viejos."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Primera corrida: escribir outputs y métricas
    store.create(run_id)
    (store.outputs_dir(run_id) / "old_output.glb").write_bytes(b"old")
    store.write_metrics(run_id, {"duration_s": 1.0})
    assert store.exists(run_id)

    # Reintento: create debe limpiar outputs viejos (porque ya existe metrics válido)
    store.create(run_id)
    assert not (store._dir(run_id) / "outputs" / "old_output.glb").exists()

    # Escribir nuevos outputs
    (store.outputs_dir(run_id) / "new_output.glb").write_bytes(b"new")
    store.write_metrics(run_id, {"duration_s": 2.0})

    # Verificar que solo existe el nuevo output
    artifacts = store.load_artifacts(run_id)
    assert "new_output.glb" in artifacts.files
    assert "old_output.glb" not in artifacts.files
    assert artifacts.metrics["duration_s"] == 2.0
