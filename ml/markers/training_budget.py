# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Canonical fail-closed training-budget enforcement for marker repairs."""

from __future__ import annotations

import json
from pathlib import Path

from ml.markers.gate_seal import require_committed_sources


CANONICAL_LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")


def require_training_budget(repo_root: Path, *, task: str, revision: str) -> None:
    """Refuse exhausted or unregistered revisions before any training output is created."""

    ledger_path = repo_root / CANONICAL_LEDGER_PATH
    if not ledger_path.is_file():
        raise RuntimeError(f"Canonical marker training-budget ledger is missing: {ledger_path}")
    require_committed_sources(repo_root, (CANONICAL_LEDGER_PATH,))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    match = next(
        (
            entry
            for entry in ledger.get("revisions", [])
            if entry.get("task") == task and entry.get("revision") == revision
        ),
        None,
    )
    if match is None:
        raise RuntimeError(f"Marker training revision is not preregistered: {task}/{revision}")
    if match.get("status") != "available":
        consumed = ", ".join(str(item) for item in match.get("consumed_candidate_ids", []))
        raise RuntimeError(
            f"Marker training budget is {match.get('status')}: {task}/{revision}; consumed candidates: {consumed}"
        )
    raise RuntimeError(
        f"Marker training revision is recorded as available but no authorized runner is bound: {task}/{revision}"
    )


__all__ = ["CANONICAL_LEDGER_PATH", "require_training_budget"]
