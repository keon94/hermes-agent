"""Generic custom job boundary used by domain runtimes."""
from __future__ import annotations

from typing import Any, Callable

from custom.core.delivery import dispatch_manifest
from custom_runtime import WorkerContext


class Job:
    """Own worker construction and manifest handling for one custom run."""

    def __init__(self, job: dict, worker_context: WorkerContext):
        self.job = job
        self._worker_context = worker_context

    def make_worker(self, worker: Any, worker_number: int) -> tuple[Any, dict, str]:
        context = self._worker_context
        worker_job = dict(self.job)
        worker_job["model"] = worker.model
        worker_job["reasoning_effort"] = worker.reasoning
        worker_jc = context.make_cron_job_config(
            context.cron_job_config.cfg, worker.model, context.cron_job_config.model_cfg,
            context.cron_job_config.cron_default_provider)
        worker_setup = context.resolve_setup(worker_job, context.job_id, context.job_name, worker_jc)
        if worker_setup.blocked is not None:
            raise RuntimeError(f"Custom worker blocked: {worker_setup.blocked[3]}")
        worker_session = f"{context.cron_session_id}_worker_{worker_number}"
        worker_agent = context.construct_agent(
            context.ai_agent_type, worker_job, context.config, worker_setup,
            workdir=context.workdir, session_id=worker_session, session_db=context.session_db)
        return worker_agent, worker_job, worker_session

    def handle_manifest(
        self,
        manifest: dict,
        *,
        adapters: Any,
        loop: Any,
        send: Callable[..., str | None],
    ) -> str | None:
        """Dispatch records through Hermes while retaining the scheduler's transport authority."""
        deliver = str(self.job.get("deliver") or "").strip().lower()
        if not deliver or deliver == "local":
            return "custom delivery manifest has no configured Hermes delivery lane"
        return dispatch_manifest(
            manifest,
            lambda record_id, content: send(
                dict(self.job, execution_id=f"{self.job.get('execution_id', self.job['id'])}:{record_id}"),
                content, adapters=adapters, loop=loop),
        )
