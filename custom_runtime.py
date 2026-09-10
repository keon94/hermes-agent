"""Typed contract between Hermes cron and locally owned custom runtimes."""
from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol


@dataclass
class WorkerContext:
    job: dict
    job_id: str
    job_name: str
    cron_session_id: str
    config: dict
    cron_job_config: Any
    workdir: Any
    session_db: Any
    make_cron_job_config: Callable[..., Any]
    resolve_setup: Callable[..., Any]
    construct_agent: Callable[..., Any]
    ai_agent_type: Any


@dataclass
class RuntimeContext:
    job: dict
    job_id: str
    job_name: str
    prompt: str
    controller_agent: Any
    ai_agent_type: Any
    config: dict
    setup: Any
    workdir: Any
    session_db: Any
    session_id: str
    task_id: str
    cancel_event: Any
    run_agent: Callable[..., dict]
    final_response: Callable[..., str]
    worker_context: WorkerContext
    teardown: Callable[..., None]


class RuntimeJob(Protocol):
    def make_worker(self, worker: Any, worker_number: int) -> tuple[Any, dict, str]: ...
    def handle_manifest(self, manifest: dict, *, adapters: Any, loop: Any, send: Callable[..., str | None]) -> str | None: ...


@dataclass(frozen=True)
class CustomRunResult:
    result: dict
    final_response: str
    job: RuntimeJob | None = None
    delivery_manifest: dict | None = None


class Runtime(ABC):
    @abstractmethod
    def handles(self, job: dict) -> bool: ...

    @abstractmethod
    def run(self, context: RuntimeContext) -> CustomRunResult: ...
