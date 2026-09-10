"""Repository-policy runtime for Job Finder; no Hermes scheduler imports."""
from __future__ import annotations

import json
import re
from typing import Any

from custom.core.execution import Phase, run_policy_workers
from custom.core.policy import (
    PolicyError,
    WorkerLedger,
    load_runtime_policy,
    parse_delivery_manifest,
    write_worker_record,
)
from custom.hooks.job import Job
from custom_runtime import CustomRunResult, Runtime, RuntimeContext


class JobFinderRuntime(Runtime):
    """Orchestrate a repository-defined workflow through policy-selected workers."""

    def handles(self, job: dict) -> bool:
        return job.get("custom_runtime") == "job-finder"

    def _settings(self, job: dict) -> dict:
        settings = job.get("custom_runtime_config")
        if not isinstance(settings, dict):
            raise PolicyError("custom_runtime_config must be a mapping")
        for key in ("repository_root", "policy_path", "runtime_dir"):
            if not isinstance(settings.get(key), str) or not settings[key].strip():
                raise PolicyError(f"custom_runtime_config requires {key}")
        return settings

    @staticmethod
    def _plan(response: str, allowed: set[str]) -> list[Phase]:
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL | re.IGNORECASE)
        try:
            doc = json.loads(match.group(1) if match else response.strip())
        except json.JSONDecodeError as exc:
            raise PolicyError("Controller did not return a JSON phase plan") from exc
        phases = doc.get("phases") if isinstance(doc, dict) else None
        if not isinstance(phases, list) or not phases or len(phases) > 32:
            raise PolicyError("Controller phase plan must contain 1-32 phases")
        result = []
        for item in phases:
            if not isinstance(item, dict) or item.get("phase") not in allowed:
                raise PolicyError("Controller selected an unauthorized phase")
            objective = item.get("objective")
            if not isinstance(objective, str) or not objective.strip() or len(objective) > 4000:
                raise PolicyError("Controller phase objective is invalid")
            result.append(Phase(item["phase"], objective.strip()))
        return result

    def run(self, context: RuntimeContext) -> CustomRunResult:
        settings = self._settings(context.job)
        policy = load_runtime_policy(settings["repository_root"], settings["policy_path"])
        allowed = {phase for worker in policy.workers for phase in worker.phases}
        planner_prompt = (
            "You are a lightweight workflow controller. Do not use tools or perform work. Return only JSON "
            "in a json fence: {\"phases\":[{\"phase\":...,\"objective\":...}]}. Select only from: "
            f"{', '.join(sorted(allowed))}. Keep objectives bounded and verifiable.\n\n{context.prompt}"
        )
        controller_result = context.run_agent(
            context.controller_agent, planner_prompt, context.job, context.job_id,
            context.job_name, context.task_id, context.cancel_event)
        controller_response = context.final_response(
            controller_result, context.job_id, context.job_name, context.ai_agent_type)
        phases = self._plan(controller_response, allowed)
        ledger = WorkerLedger(policy)
        job = Job(context.job, context.worker_context)
        run_id = str(context.job.get("execution_id") or context.session_id)
        worker_count = 0

        def execute(worker, phase, handoff):
            nonlocal worker_count
            worker_count += 1
            worker_agent, worker_job, worker_session = job.make_worker(worker, worker_count)
            worker_prompt = (
                "You are a policy-selected worker. Perform only the bounded phase below using the repository "
                "instructions and authorized tools. Return a concise, evidence-backed handoff.\n\n"
                f"Phase: {phase.name}\nObjective: {phase.objective}\n"
                f"Upstream handoff:\n{handoff}\n\nOriginal request:\n{context.prompt}"
            )
            try:
                result = context.run_agent(
                    worker_agent, worker_prompt, worker_job, context.job_id,
                    context.job_name, f"{context.task_id}:worker:{worker_count}", context.cancel_event)
                response = context.final_response(
                    result, context.job_id, context.job_name, context.ai_agent_type)
                write_worker_record(
                    settings["runtime_dir"], run_id=run_id,
                    phase_id=f"{worker_count:02d}-{phase.name}", session_id=worker_session,
                    worker=worker, policy=policy, status="completed", started_at="",
                    ended_at="", upstream_handoff={"text": handoff} if handoff else None,
                    result=result, handoff={"text": response})
                return dict(result, final_response=response)
            finally:
                context.teardown(worker_agent, context.job_id)

        results = run_policy_workers(policy, phases, execute=execute, ledger=ledger)
        final = results[-1]["response"]
        manifest = parse_delivery_manifest(final)
        return CustomRunResult(
            result={"completed": True, "total_tokens": sum(x["tokens"] for x in results),
                    "policy_commit": policy.repository_commit, "policy_sha256": policy.policy_sha256,
                    "worker_ledger": ledger.summary(), "worker_results": results},
            final_response=final, job=job, delivery_manifest=manifest)
