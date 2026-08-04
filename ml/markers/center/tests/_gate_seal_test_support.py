# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Private test-only fixture for a committed canonical gate repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping

from ml.markers.gate_seal import canonical_json_bytes, source_bundle_sha256


def _create_committed_gate_repo(
    root: Path,
    *,
    gate_config: Mapping[str, object],
    manifest_sha256: str,
    task: str = "marker-center",
    revision: str = "test-v1",
    candidate_hash_keys: tuple[str, ...] = ("onnx_sha256",),
) -> tuple[Path, Path]:
    source = root / "ml/markers/center/evaluator.py"
    split = root / "ml/markers/center/gates/test.json"
    retired = root / "ml/markers/gate-seals/retired-historical-pairs.json"
    source.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    retired.parent.mkdir(parents=True)
    source.write_text("gate = 1\n", encoding="utf-8")
    retired.write_text(json.dumps({"schema_version": 1, "pairs": []}, indent=2) + "\n", encoding="utf-8")
    gate_config_sha256 = hashlib.sha256(canonical_json_bytes(dict(gate_config))).hexdigest()
    evaluator_sha256 = source_bundle_sha256(root, (source.relative_to(root),))
    split.write_text(
        json.dumps(
            {
                "task": task,
                "revision": revision,
                "expected_candidate_hash_keys": list(candidate_hash_keys),
                "expected_dataset_manifest_sha256": manifest_sha256,
                "expected_evaluator_source_bundle_sha256": evaluator_sha256,
                "expected_gate_config_sha256": gate_config_sha256,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Marker Gate Test"),
        ("git", "config", "user.email", "marker-gate-test@example.invalid"),
        ("git", "add", "ml/markers/center/evaluator.py", "ml/markers/center/gates/test.json", "ml/markers/gate-seals/retired-historical-pairs.json"),
        ("git", "commit", "-q", "-m", "Initialize gate fixture"),
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return source, split


__all__: list[str] = []


def _create_committed_training_budget_repo(root: Path, *, status: str = "exhausted") -> Path:
    ledger = root / "ml/markers/training-budgets/production-repair-v1.json"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "ledger_mode": "canonical_repository",
                "revisions": [
                    {
                        "task": "marker-center",
                        "revision": "marker-center-production-repair-v1",
                        "status": status,
                        "consumed_candidate_ids": ["P1", "P2", "P3"],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    commands = (
        ("git", "init", "-q"),
        ("git", "config", "user.name", "Marker Budget Test"),
        ("git", "config", "user.email", "marker-budget-test@example.invalid"),
        ("git", "add", "ml/markers/training-budgets/production-repair-v1.json"),
        ("git", "commit", "-q", "-m", "Initialize training budget fixture"),
    )
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    return ledger
