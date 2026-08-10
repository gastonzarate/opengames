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
