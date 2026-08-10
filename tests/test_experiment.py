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
