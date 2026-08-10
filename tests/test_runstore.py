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

    # Escribir un metrics.json válido con flujo normal
    store.create(run_id)
    store.write_job(run_id, job)
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    # Truncar el archivo (simular interrupción a mitad de escritura)
    metrics_path = store._dir(run_id) / "metrics.json"
    metrics_path.write_text("{")  # JSON inválido

    # Debe retornar False porque el JSON es corrupto
    assert not store.exists(run_id)


def test_load_artifacts_on_nonexistent_run_raises(tmp_path, spec, image):
    """load_artifacts sobre un run_id inexistente lanza FileNotFoundError."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Directorio no existe
    assert not (store._dir(run_id)).exists()

    # Cargar artifacts sobre un run_id que nunca fue iniciado debe lanzar excepción
    with pytest.raises(FileNotFoundError):
        store.load_artifacts(run_id)

    # El directorio raíz no fue creado
    assert not (store._dir(run_id)).exists()


def test_retry_cleans_old_outputs_from_failed_run(tmp_path, spec, image):
    """Reintento sobre run_id que crasheó a mitad limpia outputs parciales.

    Usa dos instancias de RunStore para simular un nuevo proceso (crash → reintento).
    """
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Primer proceso: simular una corrida que crashea después de escribir un output parcial
    store1 = RunStore(tmp_path / "runs")
    store1.create(run_id)
    store1.write_job(run_id, job)  # Crea el marcador .in-progress
    (store1.outputs_dir(run_id) / "partial_output.glb").write_bytes(b"incomplete")
    # La corrida nunca llegó a write_metrics(), así que .in-progress sigue presente

    assert not store1.exists(run_id)  # No cacheada (crash)
    assert (store1._dir(run_id) / "outputs" / "partial_output.glb").exists()

    # Nuevo proceso (reintento): write_job() limpia outputs viejos del crash
    store2 = RunStore(tmp_path / "runs")  # Nueva instancia, _initiated_run_ids vacío
    store2.write_job(run_id, job)
    assert not (store2._dir(run_id) / "outputs" / "partial_output.glb").exists()

    # Escribir nuevos outputs
    (store2.outputs_dir(run_id) / "new_output.glb").write_bytes(b"new")
    store2.write_metrics(run_id, {"duration_s": 1.0})

    # Verificar que solo existe el nuevo output, no los restos del crash
    artifacts = store2.load_artifacts(run_id)
    assert "new_output.glb" in artifacts.files
    assert "partial_output.glb" not in artifacts.files
    assert artifacts.metrics["duration_s"] == 1.0


def test_completed_run_outputs_survive_subsequent_create(tmp_path, spec, image):
    """Regresión Critical: una corrida completada no pierde outputs por create() posterior."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Corrida completa
    store.create(run_id)
    store.write_job(run_id, job)
    (store.outputs_dir(run_id) / "sample.glb").write_bytes(b"artefacto")
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    # Verificar que el output existe
    assert (store._dir(run_id) / "outputs" / "sample.glb").exists()
    original_files = list((store._dir(run_id) / "outputs").iterdir())

    # Aceso posterior (simular orquestador que llama create() para copiar output)
    store.create(run_id)

    # Los outputs deben seguir existiendo
    assert store.exists(run_id)
    assert (store._dir(run_id) / "outputs" / "sample.glb").exists()
    artifacts = store.load_artifacts(run_id)
    assert "sample.glb" in artifacts.files
    assert list((store._dir(run_id) / "outputs").iterdir()) == original_files


def test_completed_run_survives_outputs_dir_access(tmp_path, spec, image):
    """Una corrida completada no pierde outputs cuando se accede a outputs_dir()."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Corrida completa
    store.create(run_id)
    store.write_job(run_id, job)
    (store.outputs_dir(run_id) / "sample.glb").write_bytes(b"artefacto")
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    # Acceso a outputs_dir (que internamente llama create())
    outputs_path = store.outputs_dir(run_id)

    # Los outputs deben seguir existiendo
    assert store.exists(run_id)
    assert (outputs_path / "sample.glb").exists()
    assert (outputs_path / "sample.glb").read_bytes() == b"artefacto"


def test_write_job_called_twice_in_same_attempt_preserves_outputs(tmp_path, spec, image):
    """write_job() llamado dos veces en el mismo intento no destruye outputs."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    # Primera llamada a write_job()
    store.create(run_id)
    store.write_job(run_id, job)

    # Escribir outputs después de la primera write_job()
    (store.outputs_dir(run_id) / "output1.glb").write_bytes(b"output1")

    # Segunda llamada a write_job() (ej: wrapper de reintento, reescribir job.json)
    store.write_job(run_id, job)

    # Los outputs deben sobrevivir a la segunda write_job()
    assert (store._dir(run_id) / "outputs" / "output1.glb").exists()
    assert (store._dir(run_id) / "outputs" / "output1.glb").read_bytes() == b"output1"

    # Escribir más outputs después de la segunda write_job()
    (store.outputs_dir(run_id) / "output2.glb").write_bytes(b"output2")

    # Completar la corrida
    store.write_metrics(run_id, {"duration_s": 1.0})

    # Verificar que ambos outputs existen
    artifacts = store.load_artifacts(run_id)
    assert "output1.glb" in artifacts.files
    assert "output2.glb" in artifacts.files
    assert artifacts.metrics["duration_s"] == 1.0


def test_write_job_twice_after_crash_cleans_old_outputs_on_new_process(tmp_path, spec, image):
    """Crash y reintento en nuevo proceso: write_job() limpia basura del intento anterior."""
    # Primer proceso: simular crash
    store1 = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store1.create(run_id)
    store1.write_job(run_id, job)
    (store1.outputs_dir(run_id) / "partial.glb").write_bytes(b"incomplete")
    # Crash: nunca llegamos a write_metrics()

    # Verificar que no está cacheado
    assert not store1.exists(run_id)
    assert (store1._dir(run_id) / "outputs" / "partial.glb").exists()

    # Nuevo proceso: reintento
    store2 = RunStore(tmp_path / "runs")  # Nueva instancia, _initiated_run_ids vacío

    # Primera write_job() en el nuevo proceso ve la basura del crash anterior
    # y la limpia
    store2.write_job(run_id, job)
    assert not (store2._dir(run_id) / "outputs" / "partial.glb").exists()

    # Escribir outputs limpios
    (store2.outputs_dir(run_id) / "clean.glb").write_bytes(b"clean")
    store2.write_metrics(run_id, {"duration_s": 1.0})

    # Verificar que solo existe el output limpio
    artifacts = store2.load_artifacts(run_id)
    assert "clean.glb" in artifacts.files
    assert "partial.glb" not in artifacts.files
