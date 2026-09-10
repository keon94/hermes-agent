"""Repository-policy runtime for Job Finder; no Hermes scheduler imports."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from custom.core.execution import BudgetExceeded, Phase, run_policy_workers
from custom.core.policy import (
    PolicyError,
    WorkerLedger,
    load_runtime_policy,
    parse_delivery_manifest,
    write_worker_record,
)
from custom.hooks.job import Job
from custom_runtime import CustomRunResult, Runtime, RuntimeContext


logger = logging.getLogger(__name__)


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
    def _markdown_handoff(response: str) -> str:
        """Render JSON manifests as Markdown before persisting or passing handoffs downstream."""
        text = str(response or "").strip()
        if not text:
            return ""
        try:
            manifest = parse_delivery_manifest(text)
        except PolicyError:
            return text
        sections = [card["content"] for card in manifest["role_cards"]]
        sections.append(manifest["run_summary"]["content"])
        return "\n\n---\n\n".join(section.strip() for section in sections if section.strip())

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

    def _controller_manifest(
        self,
        context: RuntimeContext,
        latest_handoff: str,
        stop_reason: str,
    ) -> str:
        prompt = (
            "You are the lightweight Job Finder controller. Do not use tools. Convert the latest "
            "verified worker handoff into the final JSON delivery manifest so notifications can be "
            "sent even though worker token budget is exhausted. Preserve every role card already "
            "found or written to Drive in the handoff. Add one run_summary.content explaining that the "
            "workflow stopped before all planned phases because of token-budget exhaustion. Return only "
            "JSON in this exact shape: {\"telegram_manifest\":{\"role_cards\":[{\"card_id\":\"...\","
            "\"content\":\"...\"}],\"run_summary\":{\"content\":\"...\"}}}. Do not include target, "
            "platform, chat_id, or thread_id.\n\n"
            f"Stop reason: {stop_reason}\n\nLatest handoff:\n{latest_handoff}"
        )
        logger.info(
            "Job Finder controller formatting manifest: job_id=%s controller_model=%s stop_reason=%s handoff_chars=%s",
            context.job_id, context.job.get("model"), stop_reason, len(latest_handoff),
        )
        result = context.run_agent(
            context.controller_agent, prompt, context.job, context.job_id,
            context.job_name, context.task_id, context.cancel_event)
        return context.final_response(result, context.job_id, context.job_name, context.ai_agent_type)

    def run(self, context: RuntimeContext) -> CustomRunResult:
        settings = self._settings(context.job)
        policy = load_runtime_policy(settings["repository_root"], settings["policy_path"])
        logger.info(
            "Job Finder runtime start: job_id=%s controller_model=%s controller_provider=%s policy_commit=%s policy_sha256=%s workers=%s",
            context.job_id, context.job.get("model"), context.job.get("provider"),
            policy.repository_commit, policy.policy_sha256,
            [(worker.model, worker.reasoning, worker.token_cap, worker.phases) for worker in policy.workers],
        )
        allowed = {phase for worker in policy.workers for phase in worker.phases}
        planner_prompt = (
            "You are a lightweight workflow controller. Do not use tools or perform work. Return only JSON "
            "in a json fence: {\"phases\":[{\"phase\":...,\"objective\":...}]}. Select only from: "
            f"{', '.join(sorted(allowed))}. Keep objectives bounded and verifiable.\n\n{context.prompt}"
        )
        logger.info(
            "Job Finder controller planning: job_id=%s controller_model=%s allowed_phases=%s",
            context.job_id, context.job.get("model"), sorted(allowed),
        )
        controller_result = context.run_agent(
            context.controller_agent, planner_prompt, context.job, context.job_id,
            context.job_name, context.task_id, context.cancel_event)
        controller_response = context.final_response(
            controller_result, context.job_id, context.job_name, context.ai_agent_type)
        phases = self._plan(controller_response, allowed)
        logger.info(
            "Job Finder controller planned phases: job_id=%s phases=%s",
            context.job_id, [(phase.name, phase.objective[:160]) for phase in phases],
        )
        ledger = WorkerLedger(policy)
        job = Job(context.job, context.worker_context)
        run_id = str(context.job.get("execution_id") or context.session_id)
        worker_count = 0

        def execute(worker, phase, handoff):
            nonlocal worker_count
            worker_count += 1
            worker_agent, worker_job, worker_session = job.make_worker(worker, worker_count)
            logger.info(
                "Job Finder worker start: job_id=%s worker_number=%s phase=%s model=%s reasoning=%s token_cap=%s session_id=%s",
                context.job_id, worker_count, phase.name, worker.model, worker.reasoning,
                worker.token_cap, worker_session,
            )
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
                raw_response = context.final_response(
                    result, context.job_id, context.job_name, context.ai_agent_type)
                response = self._markdown_handoff(raw_response)
                write_worker_record(
                    settings["runtime_dir"], run_id=run_id,
                    phase_id=f"{worker_count:02d}-{phase.name}", session_id=worker_session,
                    worker=worker, policy=policy, status="completed", started_at="",
                    ended_at="", upstream_handoff={"text": handoff} if handoff else None,
                    result=result, handoff={"text": response})
                logger.info(
                    "Job Finder worker completed: job_id=%s worker_number=%s phase=%s model=%s reasoning=%s total_tokens=%s response_format=%s",
                    context.job_id, worker_count, phase.name, worker.model, worker.reasoning,
                    result.get("total_tokens"), "manifest-json" if raw_response != response else "markdown",
                )
                return dict(result, final_response=response, raw_response=raw_response)
            finally:
                context.teardown(worker_agent, context.job_id)

        stopped_reason = None
        try:
            results = run_policy_workers(policy, phases, execute=execute, ledger=ledger)
            final = str(results[-1].get("raw_response") or results[-1]["response"])
            completed = True
        except BudgetExceeded as exc:
            results = exc.results
            stopped_reason = str(exc)
            logger.info(
                "Job Finder worker budget exhausted: job_id=%s reason=%s controller_model=%s preserved_handoff_chars=%s",
                context.job_id, stopped_reason, context.job.get("model"), len(exc.latest_response),
            )
            final = self._controller_manifest(context, exc.latest_response, stopped_reason)
            completed = False
        manifest = parse_delivery_manifest(final)
        rendered_final = self._markdown_handoff(final)
        logger.info(
            "Job Finder runtime finished: job_id=%s completed=%s controller_model=%s role_cards=%s summary_chars=%s worker_ledger=%s",
            context.job_id, completed, context.job.get("model"), len(manifest["role_cards"]),
            len(manifest["run_summary"]["content"]), ledger.summary(),
        )
        return CustomRunResult(
            result={"completed": completed, "stopped_reason": stopped_reason,
                    "total_tokens": sum(x["tokens"] for x in results),
                    "policy_commit": policy.repository_commit, "policy_sha256": policy.policy_sha256,
                    "worker_ledger": ledger.summary(), "worker_results": results},
            final_response=rendered_final, job=job, delivery_manifest=manifest)
