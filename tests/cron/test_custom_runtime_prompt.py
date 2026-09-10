from __future__ import annotations

from types import SimpleNamespace

from custom_runtime import CustomRunResult


class _Runtime:
    def __init__(self):
        self.seen_prompt = None

    def handles(self, job: dict) -> bool:
        return True

    def run(self, context):
        self.seen_prompt = context.prompt
        return CustomRunResult(result={"total_tokens": 0}, final_response="clean response")


def test_custom_runtime_receives_and_logs_clean_prompt(monkeypatch, tmp_path):
    from cron import scheduler

    runtime = _Runtime()
    job = {
        "id": "custom-prompt-job",
        "name": "Custom Prompt Job",
        "prompt": "Run the custom workflow.",
        "custom_runtime": "example",
        "skills": ["agent-browser"],
    }
    expanded_prompt = (
        '[IMPORTANT: The user has invoked the "agent-browser" skill, indicating they want you to follow its instructions.]\n'
        '---\nname: agent-browser\n---\n# giant skill body\n\nRun the custom workflow.'
    )

    monkeypatch.setattr(scheduler, "_prepare_job_prompt", lambda *args: (None, expanded_prompt))
    monkeypatch.setattr(scheduler, "get_runtime", lambda _job: runtime)
    monkeypatch.setattr(scheduler, "_load_cron_job_config", lambda *args: SimpleNamespace(cfg={}, model="controller", model_cfg={}, cron_default_provider="test"))
    monkeypatch.setattr(scheduler, "_resolve_cron_agent_setup", lambda *args: SimpleNamespace(blocked=None, model="controller"))
    monkeypatch.setattr(scheduler, "_open_cron_session_db", lambda _job: None)
    monkeypatch.setattr(scheduler, "_construct_cron_agent", lambda *args, **kwargs: object())
    monkeypatch.setattr(scheduler, "_finalize_cron_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "_teardown_cron_agent", lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "_reload_dotenv_and_publish_delivery_target", lambda _job: None)
    monkeypatch.setattr(scheduler, "_write_usage_audit", lambda _record: None)

    success, output, final_response, error = scheduler.run_job(job)

    assert success is True
    assert error is None
    assert final_response == "clean response"
    assert runtime.seen_prompt == "Run the custom workflow."
    assert "Run the custom workflow." in output
    assert "giant skill body" not in output
    assert "IMPORTANT: The user has invoked" not in output
