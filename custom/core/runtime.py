"""Custom runtime protocol and scheduler callback context."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass
class RuntimeContext:
    job: dict
    job_id: str
    job_name: str
    prompt: str
    controller_agent: Any
    ai_agent_type: Any
    config: Any
    setup: Any
    workdir: Any
    session_db: Any
    session_id: str
    task_id: str
    cancel_event: Any
    run_agent: Callable[..., dict]
    final_response: Callable[..., str]
    make_worker: Callable[..., tuple[Any, Any, str]]
    teardown: Callable[..., None]


@dataclass(frozen=True)
class CustomRunResult:
    result: dict
    final_response: str
    delivery_manifest: dict | None = None


class CustomRuntime(Protocol):
    def handles(self, job: dict) -> bool: ...
    def run(self, context: RuntimeContext) -> CustomRunResult: ...


class CustomDelivery(Protocol):
    def deliver(self, job: dict, manifest: dict, *, adapters: Any, loop: Any) -> str | None: ...
