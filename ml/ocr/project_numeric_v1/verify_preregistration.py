# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Verify frozen protocol, split fingerprints, source hashes, and Git boundary."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from .dataset import build_split, split_fingerprint
from .protocol import ProtocolViolation, SPLITS, validate_frozen_protocol

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BINDING_PATH = Path(__file__).with_name("SOURCE_BINDING.json")
REQUIRED_SOURCE_ROLES = {
    "package_init": "ml/ocr/project_numeric_v1/__init__.py",
    "protocol": "ml/ocr/project_numeric_v1/protocol.py",
    "frozen_protocol": "ml/ocr/project_numeric_v1/FROZEN_PROTOCOL.json",
    "dataset": "ml/ocr/project_numeric_v1/dataset.py",
    "model": "ml/ocr/project_numeric_v1/model.py",
    "training_entrypoint": "ml/ocr/project_numeric_v1/train.py",
    "binding_verifier": "ml/ocr/project_numeric_v1/verify_preregistration.py",
    "experiment_plan": "ml/ocr/project_numeric_v1/EXPERIMENT_PLAN.md",
    "readme": "ml/ocr/project_numeric_v1/README.md",
    "tests": "ml/ocr/project_numeric_v1/tests/test_project_numeric_v1.py",
    "audit_record": "models/manifest/ocr/PROJECT_NUMERIC_V1_PREREGISTRATION.md",
}


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _run_git(arguments: list[str], root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def verify_source_binding(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    frozen = validate_frozen_protocol()
    binding = json.loads(SOURCE_BINDING_PATH.read_text(encoding="utf-8"))
    if binding.get("binding_version") != 1:
        raise ProtocolViolation("Project-numeric source binding version is invalid.")
    sources = binding.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_SOURCE_ROLES):
        raise ProtocolViolation("Project-numeric source binding roles are incomplete.")
    for role, relative in REQUIRED_SOURCE_ROLES.items():
        record = sources[role]
        path = root / relative
        if record.get("path") != relative:
            raise ProtocolViolation(f"Source binding path mismatch for {role}.")
        if not path.is_file():
            raise ProtocolViolation(f"Bound source is missing for {role}.")
        if record.get("bytes") != path.stat().st_size:
            raise ProtocolViolation(f"Source binding byte count mismatch for {role}.")
        if record.get("sha256") != _sha256(path):
            raise ProtocolViolation(f"Source binding SHA-256 mismatch for {role}.")

    measured_fingerprints = {}
    for registration in SPLITS:
        samples = build_split(registration.split)  # type: ignore[arg-type]
        measured_fingerprints[registration.split] = split_fingerprint(samples)
    if measured_fingerprints != frozen["split_fingerprints"]:
        raise ProtocolViolation("Frozen procedural split fingerprints do not match.")
    if binding.get("split_fingerprints") != measured_fingerprints:
        raise ProtocolViolation("Source binding split fingerprints do not match.")
    return {
        "binding_valid": True,
        "source_count": len(sources),
        "split_fingerprints": measured_fingerprints,
    }


def verify_committed_preregistration(root: Path = REPOSITORY_ROOT) -> str:
    branch = _run_git(["branch", "--show-current"], root)
    if branch.returncode != 0 or branch.stdout.strip() != "main":
        raise ProtocolViolation("Training requires the committed main branch preregistration.")
    head = _run_git(["rev-parse", "HEAD"], root)
    if head.returncode != 0 or len(head.stdout.strip()) != 40:
        raise ProtocolViolation("Training requires a resolvable preregistration commit.")

    committed_paths = [*REQUIRED_SOURCE_ROLES.values(), "ml/ocr/project_numeric_v1/SOURCE_BINDING.json"]
    for relative in committed_paths:
        exists = _run_git(["cat-file", "-e", f"HEAD:{relative}"], root)
        if exists.returncode != 0:
            raise ProtocolViolation(f"Training source is not committed at HEAD: {relative}")
    status = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *committed_paths],
        root,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ProtocolViolation(
            "Training requires an unchanged committed preregistration source bundle."
        )
    verification = verify_source_binding(root)
    if verification["binding_valid"] is not True:
        raise ProtocolViolation("Training source binding did not verify.")
    return head.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-committed", action="store_true")
    arguments = parser.parse_args()
    if arguments.require_committed:
        result: dict[str, object] = {
            "binding_valid": True,
            "committed_head": verify_committed_preregistration(),
        }
    else:
        result = verify_source_binding()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
