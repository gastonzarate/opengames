import json
import os
import subprocess
import sys

import pytest

import core.runstore as runstore
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


# --- Tests agregados en rondas de arreglos (no forman parte de los 7 originales) ---
#
# Cada uno cubre una de las seis propiedades del diseño "marcador con dueño
# verificable": el `.in-progress` guarda el PID de quien lo creó y solo se
# considera huérfano -y por lo tanto seguro de limpiar- si ese proceso ya no
# existe, o si el marcador no se puede parsear.


def _dead_pid() -> int:
    """PID de un proceso que ya terminó: garantía real de "muerto", no un
    número inventado que por casualidad podría coincidir con un proceso vivo."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_exists_rejects_truncated_metrics(tmp_path, spec, image):
    """metrics.json truncado o corrupto no cuenta como caché válida."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store.create(run_id)
    store.write_job(run_id, job)
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    metrics_path = store._dir(run_id) / "metrics.json"
    metrics_path.write_text("{")  # JSON inválido
    assert not store.exists(run_id)


def test_load_artifacts_on_nonexistent_run_raises(tmp_path, spec, image):
    """load_artifacts sobre un run_id que nunca existió lanza FileNotFoundError."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    assert not store._dir(run_id).exists()
    with pytest.raises(FileNotFoundError):
        store.load_artifacts(run_id)
    assert not store._dir(run_id).exists()


def test_create_is_idempotent_and_never_destroys_outputs(tmp_path, spec, image):
    """Propiedad 1: create() es idempotente y jamás destructivo.

    Un artefacto colocado a mano en outputs/ (sin pasar por write_job ni
    write_metrics, o sea sin que exista ningún marcador todavía) tiene que
    sobrevivir a cualquier cantidad de llamadas repetidas a create().
    """
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store.create(run_id)
    (store.outputs_dir(run_id) / "temprano.glb").write_bytes(b"dato")

    for _ in range(3):
        store.create(run_id)

    assert (store._dir(run_id) / "outputs" / "temprano.glb").read_bytes() == b"dato"


def test_dead_attempt_garbage_does_not_contaminate_retry(tmp_path, spec, image):
    """Propiedad 2: la basura de un intento que murió no contamina el reintento.

    Se simula un crash real escribiendo el marcador `.in-progress` con el PID
    de un proceso que efectivamente ya terminó (no una instancia nueva en el
    mismo proceso, que es un caso distinto: ver la propiedad 6). Un
    `write_job()` posterior debe reconocer el marcador como huérfano y
    limpiar los restos antes de que el nuevo intento escriba sus outputs.
    """
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    base = store.create(run_id)
    (base / "outputs" / "partial.glb").write_bytes(b"incompleto")
    (base / ".in-progress").write_text(json.dumps({"pid": _dead_pid()}))
    assert not store.exists(run_id)

    store.write_job(run_id, job)
    assert not (base / "outputs" / "partial.glb").exists()

    (store.outputs_dir(run_id) / "limpio.glb").write_bytes(b"nuevo")
    store.write_metrics(run_id, {"duration_s": 1.0})

    artifacts = store.load_artifacts(run_id)
    assert "limpio.glb" in artifacts.files
    assert "partial.glb" not in artifacts.files


def test_completed_run_survives_subsequent_access(tmp_path, spec, image):
    """Propiedad 3: una corrida completada nunca pierde sus outputs por
    accesos posteriores (create(), outputs_dir(), o un write_job() tardío)."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store.create(run_id)
    store.write_job(run_id, job)
    (store.outputs_dir(run_id) / "sample.glb").write_bytes(b"artefacto")
    store.write_metrics(run_id, {"duration_s": 1.5})
    assert store.exists(run_id)

    store.create(run_id)
    store.outputs_dir(run_id)
    store.write_job(run_id, job)  # acceso tardío, no debería tocar nada

    assert store.exists(run_id)
    artifacts = store.load_artifacts(run_id)
    assert artifacts.files["sample.glb"].read_bytes() == b"artefacto"


def test_exists_is_false_while_in_progress_marker_present(tmp_path, spec, image):
    """Propiedad 4: exists() nunca da verdadero con `.in-progress` presente,
    incluso si metrics.json ya es válido (p. ej. una corrida completándose
    justo en este instante, entre escribir metrics y borrar el marcador)."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    base = store.create(run_id)
    store.write_job(run_id, job)
    metrics_path = base / "metrics.json"
    metrics_path.write_text(json.dumps({"duration_s": 1.0}))
    assert (base / ".in-progress").is_file()

    assert not store.exists(run_id)


def test_write_job_called_twice_in_same_attempt_preserves_outputs(tmp_path, spec, image):
    """Propiedad 5: dos `write_job()` dentro del mismo intento no se pisan."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store.write_job(run_id, job)
    (store.outputs_dir(run_id) / "output1.glb").write_bytes(b"output1")

    store.write_job(run_id, job)  # p.ej. reescribir job.json tras resolver defaults
    assert (store._dir(run_id) / "outputs" / "output1.glb").exists()

    (store.outputs_dir(run_id) / "output2.glb").write_bytes(b"output2")
    store.write_metrics(run_id, {"duration_s": 1.0})

    artifacts = store.load_artifacts(run_id)
    assert "output1.glb" in artifacts.files
    assert "output2.glb" in artifacts.files


def test_two_instances_same_process_do_not_destroy_each_others_outputs(tmp_path, spec, image):
    """Propiedad 6 (el hallazgo abierto): dos `RunStore` sobre la misma raíz,
    en el mismo proceso, no se destruyen artefactos entre sí.

    Nada en la firma pública `RunStore(root: Path)` impide instanciar dos
    veces sobre la misma raíz. La segunda instancia no puede tratar el
    intento en curso de la primera como basura huérfana solo porque su
    propio estado en memoria está vacío.
    """
    root = tmp_path / "runs"
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    a = RunStore(root)
    a.create(run_id)
    a.write_job(run_id, job)
    (a.outputs_dir(run_id) / "valioso.glb").write_bytes(b"artefacto del intento en curso")

    b = RunStore(root)  # segunda instancia, mismo proceso, misma raíz
    b.write_job(run_id, job)

    assert (a.outputs_dir(run_id) / "valioso.glb").exists()
    assert (a.outputs_dir(run_id) / "valioso.glb").read_bytes() == b"artefacto del intento en curso"


def test_pid_liveness_check_treats_marker_as_alive_outside_posix(tmp_path, spec, image, monkeypatch):
    """Fuera de POSIX, `os.kill(pid, 0)` no es una consulta benigna: en
    Windows invoca `TerminateProcess` en vez de solo preguntar. Sin forma de
    consultar sin efectos secundarios, un marcador ajeno se trata como vivo
    y no se limpia -incluso si el PID que guarda corresponde a un proceso
    que ya terminó de verdad."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    base = store.create(run_id)
    (base / "outputs" / "partial.glb").write_bytes(b"incompleto")
    (base / ".in-progress").write_text(json.dumps({"pid": _dead_pid()}))

    monkeypatch.setattr(runstore.os, "name", "nt")
    store.write_job(run_id, job)

    assert (base / "outputs" / "partial.glb").exists()


def test_marker_rewrite_is_atomic_and_never_leaves_final_path_truncated(
    tmp_path, spec, image, monkeypatch
):
    """El marcador se reescribe con el mismo patrón de temporal + os.replace()
    que usa write_metrics(). Si esa reescritura falla a mitad de camino (aquí
    se simula un crash justo antes del os.replace() final), el archivo en su
    ruta definitiva tiene que seguir siendo el contenido viejo, completo y
    parseable -nunca vacío ni truncado- para que un lector concurrente jamás
    pueda confundir un intento vivo con basura huérfana."""
    store = RunStore(tmp_path / "runs")
    job = Job(model="toy", inputs={"image": image})
    run_id = compute_run_id(job, spec)

    store.write_job(run_id, job)  # primer write_job: crea el marcador
    (store.outputs_dir(run_id) / "output1.glb").write_bytes(b"output1")
    marker = store._dir(run_id) / ".in-progress"
    original_marker_content = marker.read_text()
    assert json.loads(original_marker_content)["pid"] == os.getpid()

    def _boom(*args, **kwargs):
        raise RuntimeError("crash simulado a mitad de la reescritura del marcador")

    monkeypatch.setattr(runstore.os, "replace", _boom)
    with pytest.raises(RuntimeError):
        store.write_job(run_id, job)  # segunda llamada, mismo intento: reescribe el marcador

    # El marcador nunca quedó vacío ni truncado: sigue siendo el contenido
    # completo de antes de la reescritura fallida.
    assert marker.read_text() == original_marker_content
    assert (store.outputs_dir(run_id) / "output1.glb").exists()
