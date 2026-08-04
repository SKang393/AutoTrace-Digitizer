# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Canonical fail-closed training-budget enforcement for marker repairs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from ml.markers.gate_seal import (
    canonical_json_bytes,
    require_committed_sources,
    sha256_file,
    source_bundle_sha256,
)


CANONICAL_LEDGER_PATH = Path("ml/markers/training-budgets/production-repair-v1.json")


@dataclass(frozen=True)
class TrainingAuthorization:
    directory: Path
    opened_path: Path
    binding: dict[str, object]


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


def acquire_training_candidate(
    repo_root: Path,
    *,
    task: str,
    revision: str,
    candidate_id: str,
    config_path: Path,
    runner_source_paths: Sequence[Path],
) -> TrainingAuthorization:
    ledger_path = repo_root / CANONICAL_LEDGER_PATH
    evidence_paths = (CANONICAL_LEDGER_PATH, config_path, *runner_source_paths)
    require_committed_sources(repo_root, evidence_paths)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    entry = next(
        (
            item
            for item in ledger.get("revisions", [])
            if item.get("task") == task and item.get("revision") == revision
        ),
        None,
    )
    if entry is None or entry.get("status") != "preregistered" or entry.get("execution_authorized") is not True:
        raise RuntimeError(f"Training candidate is not authorized by the canonical ledger: {task}/{revision}/{candidate_id}")
    if entry.get("preregistered_candidate_ids") != [candidate_id] or entry.get("consumed_candidate_ids") != []:
        raise RuntimeError(f"Training candidate budget is not an unused single-candidate authorization: {candidate_id}")
    config_file = repo_root / config_path
    config_sha256 = sha256_file(config_file)
    if entry.get("candidate_config_path") != config_path.as_posix() or entry.get("candidate_config_sha256") != config_sha256:
        raise RuntimeError("Training candidate configuration does not match the canonical budget ledger")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if (config.get("task"), config.get("revision"), config.get("candidate_id")) != (task, revision, candidate_id):
        raise RuntimeError("Training candidate identity does not match the preregistered configuration")
    runner_sha256 = source_bundle_sha256(repo_root, runner_source_paths)
    if config.get("expected_runner_source_bundle_sha256") != runner_sha256:
        raise RuntimeError("Training runner source bundle does not match the preregistered configuration")
    directory = repo_root / "ml" / "markers" / "training-seals" / task / revision / candidate_id
    directory.mkdir(parents=True, exist_ok=True)
    opened_path = directory / "opened.json"
    binding: dict[str, object] = {
        "task": task,
        "revision": revision,
        "candidate_id": candidate_id,
        "candidate_config_path": config_path.as_posix(),
        "candidate_config_sha256": config_sha256,
        "runner_source_paths": sorted(path.as_posix() for path in runner_source_paths),
        "runner_source_bundle_sha256": runner_sha256,
        "training_budget_ledger_sha256": sha256_file(ledger_path),
        "committed_source_enforcement": True,
    }
    opened = {
        "schema_version": 1,
        "status": "opened",
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "binding": binding,
    }
    try:
        with opened_path.open("xb") as stream:
            stream.write(canonical_json_bytes(opened))
    except FileExistsError as error:
        raise RuntimeError(f"Training candidate was already opened: {task}/{revision}/{candidate_id}") from error
    return TrainingAuthorization(directory, opened_path, binding)


def complete_training_candidate(
    authorization: TrainingAuthorization,
    *,
    status: str,
    report_sha256: str,
) -> Path:
    result_path = authorization.directory / "result.json"
    result = {
        "schema_version": 1,
        "status": status,
        "opened_sha256": sha256_file(authorization.opened_path),
        "report_sha256": report_sha256,
    }
    try:
        with result_path.open("xb") as stream:
            stream.write(canonical_json_bytes(result))
    except FileExistsError as error:
        raise RuntimeError("Training candidate result was already recorded") from error
    return result_path


__all__ = [
    "CANONICAL_LEDGER_PATH",
    "TrainingAuthorization",
    "acquire_training_candidate",
    "complete_training_candidate",
    "require_training_budget",
]
