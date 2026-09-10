"""Tests for the generic custom Job boundary."""
from types import SimpleNamespace

from custom.hooks.job import Job
from custom_runtime import WorkerContext


def _worker_context(resolve_setup, construct_agent):
    cron_config = SimpleNamespace(cfg={"base": True}, model_cfg={"provider": "test"}, cron_default_provider="test")
    return WorkerContext(
        job={"id": "job-1", "deliver": "TELEGRAM", "execution_id": "run-1"},
        job_id="job-1",
        job_name="Job",
        cron_session_id="cron-1",
        config={"runtime": True},
        cron_job_config=cron_config,
        workdir="/tmp/work",
        session_db="db",
        make_cron_job_config=lambda cfg, model, model_cfg, provider: SimpleNamespace(
            cfg=cfg, model_cfg=model_cfg, cron_default_provider=provider, model=model),
        resolve_setup=resolve_setup,
        construct_agent=construct_agent,
        ai_agent_type="agent-type",
    )


def test_make_worker_owns_worker_job_and_session_setup():
    calls = []

    def resolve_setup(job, job_id, job_name, cron_config):
        calls.append((job, job_id, job_name, cron_config))
        return SimpleNamespace(blocked=None)

    def construct_agent(agent_type, job, config, setup, **kwargs):
        calls.append((agent_type, job, config, setup, kwargs))
        return "worker-agent"

    job = Job(
        {"id": "job-1", "model": "controller", "execution_id": "run-1"},
        _worker_context(resolve_setup, construct_agent),
    )
    worker = SimpleNamespace(model="worker-model", reasoning="high")

    agent, worker_job, session = job.make_worker(worker, 2)

    assert agent == "worker-agent"
    assert worker_job["model"] == "worker-model"
    assert worker_job["reasoning_effort"] == "high"
    assert session == "cron-1_worker_2"
    assert calls[0][0]["id"] == "job-1"
    assert calls[1][0] == "agent-type"
    assert calls[1][1] == worker_job


def test_handle_manifest_uses_scheduler_sender_and_preserves_record_ids():
    sent = []
    job = Job(
        {"id": "job-1", "deliver": "telegram", "execution_id": "run-1"},
        _worker_context(lambda *args: SimpleNamespace(blocked=None), lambda *args, **kwargs: None),
    )
    manifest = {
        "role_cards": [{"card_id": "role-1", "content": "candidate"}],
        "run_summary": {"content": "done"},
    }

    error = job.handle_manifest(
        manifest,
        adapters="adapters",
        loop="loop",
        send=lambda target, content, **kwargs: sent.append((target, content, kwargs)) or None,
    )

    assert error is None
    assert [item[:2] for item in sent] == [
        ({"id": "job-1", "deliver": "telegram", "execution_id": "run-1:role-1"}, "candidate"),
        ({"id": "job-1", "deliver": "telegram", "execution_id": "run-1:run-summary"}, "done"),
    ]
    assert all(item[2] == {"adapters": "adapters", "loop": "loop"} for item in sent)
