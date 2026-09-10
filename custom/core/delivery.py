"""Transport-neutral delivery manifest dispatch."""
from __future__ import annotations

from typing import Callable


def dispatch_manifest(manifest: dict, send: Callable[[str, str], str | None]) -> str | None:
    """Send each validated record exactly once through a scheduler-owned callback."""
    records = [(card["card_id"], card["content"]) for card in manifest.get("role_cards", [])]
    records.append(("run-summary", manifest["run_summary"]["content"]))
    errors = []
    for record_id, content in records:
        error = send(record_id, content)
        if error:
            errors.append(f"{record_id}: {error}")
    return "; ".join(errors) if errors else None
