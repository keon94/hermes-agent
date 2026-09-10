"""Repository-policy cron orchestration is fail-closed and budget-accounted."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from custom.core.policy import (
    PolicyError,
    PolicyWorker,
    RuntimePolicy,
    WorkerLedger,
    load_runtime_policy,
    parse_runtime_policy,
    write_worker_record,
    parse_delivery_manifest,
)
from custom.core.execution import Phase, run_policy_workers
from custom.core.delivery import dispatch_manifest


def _document(body: str) -> str:
    return "# Policy\n\n```yaml\nruntime_policy:\n" + body + "\n```\n"


def _valid_policy() -> str:
    return _document(
        "  version: 1\n"
        "  total_token_cap: 1000\n"
        "  workers:\n"
        "    - model: terra\n"
        "      reasoning: high\n"
        "      ratio: 0.75\n"
        "      phases: [analysis, review, verification]\n"
        "    - model: luna\n"
        "      reasoning: low\n"
        "      ratio: 0.25\n"
        "      phases: [extraction]\n"
        "  verification:\n"
        "    lower_priority_result_requires_higher_priority_review: true\n"
        "  accounting:\n"
        "    fields: [input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, total_tokens, api_call_count]"
    )


def test_parse_policy_derives_allocation_from_total_and_ratio():
    policy = parse_runtime_policy(_valid_policy())

    assert policy.total_token_cap == 1000
    assert [(w.model, w.allocation) for w in policy.workers] == [("terra", 750), ("luna", 250)]
    assert policy.select_worker("extraction", {})[0].model == "luna"
    assert policy.select_worker("analysis", {})[0].model == "terra"


def test_parse_policy_rejects_redundant_worker_token_cap():
    with pytest.raises(PolicyError, match="must not define token_cap"):
        parse_runtime_policy(_valid_policy().replace("      ratio: 0.75", "      ratio: 0.75\n      token_cap: 750"))


@pytest.mark.parametrize("replacement", ["0.90", "0.0", "nope"])
def test_parse_policy_rejects_invalid_or_nonunit_ratios(replacement):
    text = _valid_policy().replace("      ratio: 0.25", f"      ratio: {replacement}")
    with pytest.raises(PolicyError):
        parse_runtime_policy(text)


def test_load_runtime_policy_records_commit_and_hash_and_refuses_dirty_checkout(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    policy_path = repo / "special/job-finder/llm-budget.md"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(_valid_policy(), encoding="utf-8")

    policy = load_runtime_policy(
        repo, "special/job-finder/llm-budget.md", git_status=lambda _: "", git_rev=lambda _: "abc1234"
    )
    assert policy.repository_commit == "abc1234"
    assert len(policy.policy_sha256) == 64

    with pytest.raises(PolicyError, match="clean"):
        load_runtime_policy(
            repo, "special/job-finder/llm-budget.md", git_status=lambda _: " M policy", git_rev=lambda _: "abc123"
        )


def test_worker_record_is_durable_and_preserves_unavailable_usage(tmp_path: Path):
    worker = PolicyWorker("luna", "high", 0.25, 250, ("extraction",))
    policy = RuntimePolicy(1, 1000, (worker,), False, (), "abc", "hash")
    record = write_worker_record(
        tmp_path,
        run_id="run-1",
        phase_id="phase-1",
        session_id="session-1",
        worker=worker,
        policy=policy,
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        upstream_handoff=None,
        result={"input_tokens": 3, "total_tokens": 7, "api_calls": 2},
        handoff={"acceptance": {"passed": True}, "output": {"x": 1}},
    )

    persisted = json.loads(Path(record).read_text(encoding="utf-8"))
    assert persisted["usage"]["input_tokens"] == 3
    assert persisted["usage"]["cached_input_tokens"] is None
    assert persisted["usage_unavailable"]["cached_input_tokens"] == "provider did not return this field"
    assert persisted["usage"]["api_call_count"] == 2
    assert persisted["handoff_sha256"]


def test_luna_work_is_followed_by_terra_review_and_ledger_limits():
    policy = parse_runtime_policy(_valid_policy())
    calls = []

    def execute(worker, phase, handoff):
        calls.append((worker.model, phase.name, handoff))
        return {"total_tokens": 10, "final_response": f"{worker.model}:{phase.name}"}

    results = run_policy_workers(
        policy, [Phase("extraction", "extract evidence"), Phase("analysis", "decide")],
        execute=execute, ledger=WorkerLedger(policy)
    )

    assert [call[:2] for call in calls] == [("luna", "extraction"), ("terra", "review"), ("terra", "analysis")]
    assert results[-1]["response"] == "terra:analysis"


def test_delivery_manifest_is_bounded_and_target_pinned():
    response = '''```json
{"telegram_manifest":{"role_cards":[{"card_id":"role-1","content":"candidate"}],"run_summary":{"content":"done"}}}
```'''
    assert parse_delivery_manifest(response)["role_cards"][0]["card_id"] == "role-1"
    with pytest.raises(PolicyError, match="target"):
        parse_delivery_manifest('{"telegram_manifest":{"target":"telegram:bad","role_cards":[],"run_summary":{"content":"done"}}}')
    with pytest.raises(PolicyError, match="unique"):
        parse_delivery_manifest('{"telegram_manifest":{"role_cards":[{"card_id":"x","content":"a"},{"card_id":"x","content":"b"}],"run_summary":{"content":"done"}}}')


def test_manifest_dispatches_cards_then_one_summary():
    sent = []
    manifest = {"role_cards": [{"card_id": "role-1", "content": "candidate"}], "run_summary": {"content": "done"}}
    assert dispatch_manifest(manifest, lambda record_id, content: sent.append((record_id, content))) is None
    assert sent == [("role-1", "candidate"), ("run-summary", "done")]
