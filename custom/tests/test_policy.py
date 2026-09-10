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
        "  version: 2\n"
        "  workers:\n"
        "    - model: terra\n"
        "      reasoning: high\n"
        "      token_cap: 0\n"
        "      phases: [analysis, review, verification]\n"
        "    - model: luna\n"
        "      reasoning: low\n"
        "      phases: [extraction]\n"
        "  verification:\n"
        "    lower_priority_result_requires_higher_priority_review: true\n"
        "  accounting:\n"
        "    fields: [input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, total_tokens, api_call_count]"
    )


def test_parse_policy_uses_individual_optional_worker_caps():
    policy = parse_runtime_policy(_valid_policy())

    assert [(w.model, w.token_cap) for w in policy.workers] == [("terra", None), ("luna", None)]
    assert policy.select_worker("extraction", {})[0].model == "luna"
    assert policy.select_worker("analysis", {})[0].model == "terra"


def test_parse_policy_accepts_positive_worker_token_cap():
    policy = parse_runtime_policy(_valid_policy().replace("      token_cap: 0", "      token_cap: 750"))
    assert policy.workers[0].token_cap == 750


def test_absent_or_zero_worker_cap_is_unlimited():
    policy = parse_runtime_policy(_valid_policy())
    ledger = WorkerLedger(policy)

    for worker in policy.workers:
        ledger.record(worker, "test", 1_000_000)

    assert ledger.summary() == {
        "terra": {"used": 1_000_000, "token_cap": None},
        "luna": {"used": 1_000_000, "token_cap": None},
    }


@pytest.mark.parametrize("replacement", ["-1", "1.5", "nope"])
def test_parse_policy_rejects_invalid_worker_token_cap(replacement):
    text = _valid_policy().replace("      token_cap: 0", f"      token_cap: {replacement}")
    with pytest.raises(PolicyError):
        parse_runtime_policy(text)


def test_parse_policy_rejects_legacy_global_cap_and_ratios():
    with pytest.raises(PolicyError, match="must be 2"):
        parse_runtime_policy(_valid_policy().replace("  version: 2", "  version: 1"))
    with pytest.raises(PolicyError, match="total_token_cap"):
        parse_runtime_policy(_valid_policy().replace("  workers:", "  total_token_cap: 1000\n  workers:"))
    with pytest.raises(PolicyError, match="ratio"):
        parse_runtime_policy(_valid_policy().replace("      token_cap: 0", "      ratio: 0.75"))


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
    worker = PolicyWorker("luna", "high", 250, ("extraction",))
    policy = RuntimePolicy(2, (worker,), False, (), "abc", "hash")
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
