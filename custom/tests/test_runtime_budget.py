"""Budget exhaustion still produces a deliverable manifest."""
from __future__ import annotations

import importlib
import logging
from types import SimpleNamespace

import pytest

from custom.core.execution import BudgetExceeded, Phase, run_policy_workers
from custom.core.policy import PolicyWorker, RuntimePolicy, WorkerLedger
from custom_runtime import RuntimeContext, WorkerContext


MANIFEST = (
    '{"telegram_manifest":{"role_cards":[{"card_id":"role-1","content":"saved to Drive"}],'
    '"run_summary":{"content":"Stopped after budget exhaustion; delivered partial results."}}}'
)
ROOT_MANIFEST = (
    '{"role_cards":[{"card_id":"role-1","content":"saved to Drive"}],'
    '"run_summary":{"content":"Stopped after budget exhaustion; delivered partial results."}}'
)


def test_budget_exhaustion_preserves_latest_worker_handoff():
    worker = PolicyWorker("luna", "low", 1.0, 10, ("extraction",))
    policy = RuntimePolicy(1, 10, (worker,), False, (), "abc", "hash")

    with pytest.raises(BudgetExceeded) as raised:
        run_policy_workers(
            policy,
            [Phase("extraction", "extract partial results")],
            execute=lambda *_: {"total_tokens": 11, "final_response": MANIFEST},
            ledger=WorkerLedger(policy),
        )

    assert raised.value.latest_response == MANIFEST
    assert raised.value.results == [
        {"worker": "luna", "phase": "extraction", "response": MANIFEST, "tokens": 11, "over_budget": True}
    ]


def test_budget_exhaustion_after_mandatory_review_preserves_reviewer_handoff():
    reviewer = PolicyWorker("terra", "high", 0.75, 10, ("review",))
    worker = PolicyWorker("luna", "low", 0.25, 100, ("extraction",))
    policy = RuntimePolicy(1, 110, (reviewer, worker), True, (), "abc", "hash", ("review",))

    def execute(selected, phase, _handoff):
        if selected.model == "terra":
            return {"total_tokens": 11, "final_response": MANIFEST}
        return {"total_tokens": 1, "final_response": "partial worker handoff"}

    with pytest.raises(BudgetExceeded) as raised:
        run_policy_workers(
            policy,
            [Phase("extraction", "extract partial results")],
            execute=execute,
            ledger=WorkerLedger(policy),
        )

    assert raised.value.latest_response == MANIFEST
    assert raised.value.results[-1] == {
        "worker": "terra", "phase": "review", "response": MANIFEST, "tokens": 11, "over_budget": True
    }


def test_job_finder_budget_exhaustion_uses_controller_for_manifest(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    runtime_module = importlib.import_module("custom.job-finder.runtime")
    worker = PolicyWorker("luna", "low", 1.0, 10, ("extraction",))
    policy = RuntimePolicy(1, 10, (worker,), False, (), "abc", "hash")
    controller_prompts = []

    monkeypatch.setattr(runtime_module, "load_runtime_policy", lambda *_: policy)
    monkeypatch.setattr(runtime_module, "write_worker_record", lambda *args, **kwargs: tmp_path / "record.json")

    def run_agent(agent, prompt, *_):
        if agent == "controller":
            controller_prompts.append(prompt)
            if "Convert the latest" in prompt:
                return {"total_tokens": 1, "final_response": MANIFEST}
            return {"total_tokens": 1, "final_response": '{"phases":[{"phase":"extraction","objective":"extract"}]}' }
        return {"total_tokens": 11, "final_response": MANIFEST}

    context = RuntimeContext(
        job={
            "id": "job-1",
            "custom_runtime": "job-finder",
            "custom_runtime_config": {
                "repository_root": str(tmp_path),
                "policy_path": "policy.md",
                "runtime_dir": str(tmp_path / "runtime"),
            },
        },
        job_id="job-1",
        job_name="Job Finder",
        prompt="run",
        controller_agent="controller",
        ai_agent_type=object,
        config={},
        setup=SimpleNamespace(),
        workdir=tmp_path,
        session_db=None,
        session_id="session-1",
        task_id="task-1",
        cancel_event=None,
        run_agent=run_agent,
        final_response=lambda result, *_: result["final_response"],
        worker_context=WorkerContext(
            job={"id": "job-1"},
            job_id="job-1",
            job_name="Job Finder",
            cron_session_id="cron-1",
            config={},
            cron_job_config=SimpleNamespace(cfg={}, model_cfg={}, cron_default_provider="test"),
            workdir=tmp_path,
            session_db=None,
            make_cron_job_config=lambda *args: SimpleNamespace(),
            resolve_setup=lambda *args: SimpleNamespace(blocked=None),
            construct_agent=lambda *args, **kwargs: "worker",
            ai_agent_type=object,
        ),
        teardown=lambda *args: None,
    )

    result = runtime_module.JobFinderRuntime().run(context)

    assert result.result["completed"] is False
    assert "exceeded its derived allocation" in result.result["stopped_reason"]
    assert result.delivery_manifest["role_cards"][0]["card_id"] == "role-1"
    assert result.final_response == "saved to Drive\n\n---\n\nStopped after budget exhaustion; delivered partial results."
    assert "\\n" not in result.final_response
    assert '"role_cards"' not in result.final_response
    assert any("Convert the latest" in prompt for prompt in controller_prompts)
    assert "controller_model=None" in caplog.text
    assert "model=luna reasoning=low" in caplog.text


def test_markdown_handoff_accepts_root_delivery_manifest():
    runtime_module = importlib.import_module("custom.job-finder.runtime")

    assert runtime_module.JobFinderRuntime._markdown_handoff(ROOT_MANIFEST) == (
        "saved to Drive\n\n---\n\nStopped after budget exhaustion; delivered partial results."
    )
