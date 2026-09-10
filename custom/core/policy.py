"""Fail-closed repository policy loading and worker-run accounting for cron.

The runtime never treats an agent prompt as policy.  A job opts in with a
repository root and policy path; before every run this module verifies that
checkout is clean, parses the committed Markdown policy, and derives each
worker's allowance exclusively from ``total_token_cap * ratio``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

import yaml


class PolicyError(RuntimeError):
    """A policy is unavailable or unsafe to use.  Runs must stop."""


_USAGE_FIELDS = (
    "input_tokens", "cached_input_tokens", "output_tokens", "reasoning_tokens",
    "total_tokens", "api_call_count",
)


@dataclass(frozen=True)
class PolicyWorker:
    model: str
    reasoning: str
    ratio: float
    allocation: int
    phases: tuple[str, ...]


@dataclass(frozen=True)
class RuntimePolicy:
    version: int
    total_token_cap: int
    workers: tuple[PolicyWorker, ...]
    higher_priority_review: bool
    accounting_fields: tuple[str, ...]
    repository_commit: str
    policy_sha256: str
    review_phases: tuple[str, ...] = ("review", "verification", "filtering", "analysis")

    def select_worker(
        self, phase: str, policy_flags: Mapping[str, Any] | None = None
    ) -> tuple[PolicyWorker, bool]:
        """Return the allowed worker and whether a higher-priority review follows."""
        del policy_flags  # reserved for future policy keys; selection remains declarative.
        matches = [worker for worker in self.workers if phase in worker.phases]
        if not matches:
            raise PolicyError(f"No worker is authorized for phase {phase!r}")
        worker = max(matches, key=lambda candidate: candidate.ratio)
        highest = max(self.workers, key=lambda candidate: candidate.ratio)
        return worker, bool(self.higher_priority_review and worker != highest)


# Short aliases keep scheduler-facing code legible while the serialized policy
# names remain explicit and backwards-compatible.
WorkerPolicy = PolicyWorker


class WorkerLedger:
    """Per-run derived-allocation accounting. A cap breach fails the whole run."""

    def __init__(self, policy: RuntimePolicy):
        self.policy = policy
        self._used = {worker.model: 0 for worker in policy.workers}
        self.records: list[dict[str, Any]] = []

    def record(self, worker: PolicyWorker, phase: str, total_tokens: int) -> None:
        if not isinstance(total_tokens, int) or total_tokens < 0:
            raise PolicyError("total_tokens must be a non-negative integer")
        used = self._used[worker.model] + total_tokens
        if used > worker.allocation:
            raise PolicyError(
                f"Worker {worker.model!r} exceeded its derived allocation "
                f"({used} > {worker.allocation})")
        self._used[worker.model] = used
        self.records.append({"worker": worker.model, "phase": phase, "total_tokens": total_tokens})

    def summary(self) -> dict[str, dict[str, int]]:
        return {
            worker.model: {"used": self._used[worker.model], "allocation": worker.allocation}
            for worker in self.policy.workers
        }


def parse_delivery_manifest(response: str) -> dict[str, Any]:
    """Validate the scheduler-owned Telegram manifest emitted by the release worker."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
    raw = match.group(1) if match else response.strip()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyError("Release worker did not return a JSON delivery manifest") from exc
    manifest = document.get("telegram_manifest") if isinstance(document, Mapping) else None
    if manifest is None and isinstance(document, Mapping) and {"role_cards", "run_summary"} <= set(document):
        manifest = document
    if not isinstance(manifest, Mapping):
        raise PolicyError("Delivery manifest must define telegram_manifest")
    if any(key in manifest for key in ("target", "platform", "chat_id", "thread_id")):
        raise PolicyError("Delivery manifest cannot override its target")
    cards = manifest.get("role_cards")
    summary = manifest.get("run_summary")
    if not isinstance(cards, list) or len(cards) > 50 or not isinstance(summary, Mapping):
        raise PolicyError("Delivery manifest needs <=50 role_cards and exactly one run_summary")
    seen: set[str] = set()
    normalized = []
    for card in cards:
        if not isinstance(card, Mapping) or not isinstance(card.get("card_id"), str) or not isinstance(card.get("content"), str):
            raise PolicyError("Each role card needs string card_id and content")
        card_id = card["card_id"].strip()
        content = card["content"].strip()
        if not card_id or card_id in seen or len(content) > 12000:
            raise PolicyError("Role card IDs must be unique and content must be bounded")
        seen.add(card_id)
        normalized.append({"card_id": card_id, "content": content})
    if not isinstance(summary.get("content"), str) or not summary["content"].strip() or len(summary["content"]) > 12000:
        raise PolicyError("run_summary.content must be bounded non-empty text")
    return {"role_cards": normalized, "run_summary": {"content": summary["content"].strip()}}


def _yaml_policy(markdown: str) -> Mapping[str, Any]:
    match = re.search(r"```ya?ml\s*\n(.*?)\n```", markdown, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        raise PolicyError("Policy Markdown must contain a fenced YAML block")
    try:
        document = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise PolicyError(f"Policy YAML is invalid: {exc}") from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("runtime_policy"), Mapping):
        raise PolicyError("Policy YAML must define a runtime_policy mapping")
    return document["runtime_policy"]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError(f"{name} must be a positive integer")
    return value


def parse_runtime_policy(
    markdown: str, *, repository_commit: str = "unresolved", policy_sha256: str = "unresolved"
) -> RuntimePolicy:
    raw = _yaml_policy(markdown)
    version = _positive_int(raw.get("version"), "runtime_policy.version")
    total = _positive_int(raw.get("total_token_cap"), "runtime_policy.total_token_cap")
    raw_workers = raw.get("workers")
    if not isinstance(raw_workers, Sequence) or isinstance(raw_workers, (str, bytes)) or not raw_workers:
        raise PolicyError("runtime_policy.workers must be a non-empty list")

    workers: list[PolicyWorker] = []
    ratios: list[float] = []
    seen_models: set[str] = set()
    for index, candidate in enumerate(raw_workers):
        if not isinstance(candidate, Mapping):
            raise PolicyError(f"workers[{index}] must be a mapping")
        if "token_cap" in candidate:
            raise PolicyError("workers must not define token_cap; allocations derive from ratio")
        model = candidate.get("model")
        reasoning = candidate.get("reasoning")
        phases = candidate.get("phases")
        ratio = candidate.get("ratio")
        if not isinstance(model, str) or not model.strip() or model in seen_models:
            raise PolicyError("Each worker needs a unique non-empty model")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise PolicyError(f"Worker {model!r} needs a reasoning value")
        if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or not 0 < float(ratio) <= 1:
            raise PolicyError(f"Worker {model!r} ratio must be in (0, 1]")
        if not isinstance(phases, Sequence) or isinstance(phases, (str, bytes)) or not phases or not all(isinstance(p, str) and p for p in phases):
            raise PolicyError(f"Worker {model!r} phases must be a non-empty string list")
        seen_models.add(model)
        ratios.append(float(ratio))
        workers.append(PolicyWorker(model.strip(), reasoning.strip(), float(ratio), 0, tuple(phases)))

    if abs(sum(ratios) - 1.0) > 0.000001:
        raise PolicyError("Worker ratios must sum exactly to 1.0")
    allocated = [int(total * worker.ratio) for worker in workers]
    # Assign the harmless rounding remainder to the largest allotted worker, preserving total.
    allocated[max(range(len(allocated)), key=lambda i: workers[i].ratio)] += total - sum(allocated)
    workers = [
        PolicyWorker(worker.model, worker.reasoning, worker.ratio, allocated[index], worker.phases)
        for index, worker in enumerate(workers)
    ]

    verification = raw.get("verification", {})
    if not isinstance(verification, Mapping):
        raise PolicyError("runtime_policy.verification must be a mapping")
    accounting = raw.get("accounting", {})
    if not isinstance(accounting, Mapping):
        raise PolicyError("runtime_policy.accounting must be a mapping")
    fields = accounting.get("fields", _USAGE_FIELDS)
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)) or not all(isinstance(field, str) for field in fields):
        raise PolicyError("runtime_policy.accounting.fields must be a string list")
    review_phases = verification.get("review_phases", ("review", "verification", "filtering", "analysis"))
    if not isinstance(review_phases, Sequence) or isinstance(review_phases, (str, bytes)) or not review_phases or not all(isinstance(phase, str) and phase for phase in review_phases):
        raise PolicyError("runtime_policy.verification.review_phases must be a non-empty string list")
    return RuntimePolicy(
        version, total, tuple(workers),
        bool(verification.get("lower_priority_result_requires_higher_priority_review", False)),
        tuple(fields), repository_commit, policy_sha256, tuple(review_phases),
    )


def _git_stdout(repository_root: Path, args: list[str]) -> str:
    process = subprocess.run(["git", "-C", str(repository_root), *args], check=False,
                             capture_output=True, text=True, timeout=30)
    if process.returncode:
        message = process.stderr.strip() or process.stdout.strip() or "unknown git failure"
        raise PolicyError(f"Cannot inspect policy repository: {message}")
    return process.stdout.strip()


def load_runtime_policy(
    repository_root: str | Path,
    policy_path: str | Path,
    *,
    git_status: Callable[[Path], str] | None = None,
    git_rev: Callable[[Path], str] | None = None,
) -> RuntimePolicy:
    root = Path(repository_root).expanduser().resolve()
    policy_relative = Path(policy_path)
    if policy_relative.is_absolute() or ".." in policy_relative.parts:
        raise PolicyError("policy_path must be repository-relative")
    if not (root / ".git").exists():
        raise PolicyError("Policy repository must be a Git checkout")
    status = (git_status or (lambda path: _git_stdout(path, ["status", "--porcelain"]))) (root)
    if status.strip():
        raise PolicyError("Policy repository must be clean before a run")
    path = (root / policy_relative).resolve()
    if root not in path.parents or not path.is_file():
        raise PolicyError("Configured policy_path does not resolve to a regular repository file")
    contents = path.read_text(encoding="utf-8")
    commit = (git_rev or (lambda repo: _git_stdout(repo, ["rev-parse", "HEAD"]))) (root)
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit):
        raise PolicyError("Policy repository did not return a valid HEAD commit")
    digest = hashlib.sha256(contents.encode("utf-8")).hexdigest()
    return parse_runtime_policy(contents, repository_commit=commit, policy_sha256=digest)


def write_worker_record(
    runtime_dir: str | Path,
    *,
    run_id: str,
    phase_id: str,
    session_id: str,
    worker: PolicyWorker,
    policy: RuntimePolicy,
    status: str,
    started_at: str,
    ended_at: str,
    upstream_handoff: Mapping[str, Any] | None,
    result: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> Path:
    """Write one immutable worker ledger row. Missing provider fields stay explicit."""
    directory = Path(runtime_dir).expanduser().resolve() / run_id
    directory.mkdir(parents=True, exist_ok=True)
    usage: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    aliases = {"api_call_count": ("api_call_count", "api_calls")}
    for field in _USAGE_FIELDS:
        value = next((result.get(key) for key in aliases.get(field, (field,)) if key in result), None)
        usage[field] = value
        if value is None:
            unavailable[field] = "provider did not return this field"
    handoff_json = json.dumps(handoff, sort_keys=True, separators=(",", ":"), default=str)
    upstream_json = json.dumps(upstream_handoff, sort_keys=True, separators=(",", ":"), default=str) if upstream_handoff else None
    payload = {
        "schema_version": 1, "run_id": run_id, "phase_id": phase_id, "session_id": session_id,
        "status": status, "started_at": started_at, "ended_at": ended_at,
        "policy": {"commit": policy.repository_commit, "sha256": policy.policy_sha256,
                   "total_token_cap": policy.total_token_cap},
        "worker": {"model": worker.model, "reasoning": worker.reasoning, "ratio": worker.ratio,
                   "allocation": worker.allocation, "phases": list(worker.phases)},
        "usage": usage, "usage_unavailable": unavailable,
        "upstream_handoff_sha256": hashlib.sha256(upstream_json.encode()).hexdigest() if upstream_json else None,
        "handoff_sha256": hashlib.sha256(handoff_json.encode()).hexdigest(),
        "handoff": handoff,
    }
    path = directory / f"{phase_id}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
