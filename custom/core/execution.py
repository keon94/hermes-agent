"""Scheduler-agnostic repository-policy worker execution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from custom.core.policy import PolicyError, WorkerLedger, WorkerPolicy


@dataclass(frozen=True)
class Phase:
    name: str
    objective: str


class BudgetExceeded(PolicyError):
    """Raised after preserving the latest worker handoff when a policy cap is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        results: list[dict[str, Any]],
        latest_response: str,
    ):
        super().__init__(message)
        self.results = results
        self.latest_response = latest_response


def _record_or_stop(
    ledger: WorkerLedger,
    worker: WorkerPolicy,
    phase: str,
    usage: int,
    response: str,
    raw_response: str | None,
    results: list[dict[str, Any]],
) -> None:
    result = {"worker": worker.model, "phase": phase, "response": response, "tokens": usage}
    if raw_response is not None:
        result["raw_response"] = raw_response
    try:
        ledger.record(worker, phase, usage)
    except PolicyError as exc:
        raise BudgetExceeded(
            str(exc), results=[*results, {**result, "over_budget": True}], latest_response=response)
    results.append(result)


def choose_worker(policy, phase: str) -> WorkerPolicy:
    """Return the policy-selected worker; ratio precedence handles intentional overlap."""
    try:
        return policy.select_worker(phase, {})[0]
    except Exception as exc:
        raise ValueError(f"Policy cannot select a worker for phase {phase!r}: {exc}") from exc


def require_review(policy, worker: WorkerPolicy) -> WorkerPolicy | None:
    """Lower-priority work requires a higher-priority reviewer before release."""
    higher = [candidate for candidate in policy.workers if candidate.ratio > worker.ratio]
    if not higher:
        return None
    reviewers = [candidate for candidate in higher if any(
        phase in candidate.phases for phase in policy.review_phases
    )]
    if not reviewers:
        raise ValueError("Policy must contain exactly one higher-priority review worker")
    return max(reviewers, key=lambda candidate: candidate.ratio)


def review_phase_name(policy, worker: WorkerPolicy) -> str:
    """Choose a declared high-priority phase suitable for reviewing a lower-priority handoff."""
    for phase in policy.review_phases:
        if phase in worker.phases:
            return phase
    raise ValueError(f"Worker {worker.model} has no declared review-capable phase")


def run_policy_workers(
    policy,
    phases: list[Phase],
    *,
    execute: Callable[[WorkerPolicy, Phase, str], dict[str, Any]],
    ledger: WorkerLedger,
) -> list[dict[str, Any]]:
    """Run declared phases and mandatory reviews; persist usage after every worker.

    ``execute`` returns an agent result containing a numeric ``total_tokens`` and
    a non-empty ``final_response``.  Missing usage fails closed because a ratio
    budget without accounting cannot be verified.
    """
    handoff = ""
    results: list[dict[str, Any]] = []
    for phase in phases:
        worker = choose_worker(policy, phase.name)
        result = execute(worker, phase, handoff)
        usage = result.get("total_tokens")
        response = str(result.get("final_response") or "").strip()
        if not isinstance(usage, int) or usage < 0:
            raise RuntimeError(f"Worker {worker.model} did not return numeric total_tokens")
        if not response:
            raise RuntimeError(f"Worker {worker.model} returned an empty handoff")
        _record_or_stop(ledger, worker, phase.name, usage, response, result.get("raw_response"), results)
        handoff = response

        reviewer = require_review(policy, worker)
        if reviewer is not None:
            review_name = review_phase_name(policy, reviewer)
            review = Phase(review_name, f"Review and verify the prior {phase.name} phase before release.")
            reviewed = execute(reviewer, review, handoff)
            review_usage = reviewed.get("total_tokens")
            review_response = str(reviewed.get("final_response") or "").strip()
            if not isinstance(review_usage, int) or review_usage < 0:
                raise RuntimeError(f"Reviewer {reviewer.model} did not return numeric total_tokens")
            if not review_response:
                raise RuntimeError(f"Reviewer {reviewer.model} returned an empty handoff")
            _record_or_stop(ledger, reviewer, review_name, review_usage, review_response, reviewed.get("raw_response"), results)
            handoff = review_response
    return results
