# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Verify Candidate 2 source binding, split seals, and committed-main boundary."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from typing import Any

from .candidate2_dataset import build_candidate2_split
from .candidate2_protocol import ProtocolViolation, validate_frozen_protocol
from .dataset import split_fingerprint

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_BINDING_PATH = Path(__file__).with_name("SOURCE_BINDING_CANDIDATE_2.json")
REQUIRED_SOURCE_ROLES = {
    "base_dataset": "ml/ocr/project_numeric_v1/dataset.py",
    "base_model": "ml/ocr/project_numeric_v1/model.py",
    "base_protocol": "ml/ocr/project_numeric_v1/protocol.py",
    "base_training": "ml/ocr/project_numeric_v1/train.py",
    "candidate2_dataset": "ml/ocr/project_numeric_v1/candidate2_dataset.py",
    "candidate2_protocol": "ml/ocr/project_numeric_v1/candidate2_protocol.py",
    "candidate2_frozen_protocol": "ml/ocr/project_numeric_v1/CANDIDATE_2_PROTOCOL.json",
    "candidate2_training": "ml/ocr/project_numeric_v1/candidate2_train.py",
    "candidate2_verifier": "ml/ocr/project_numeric_v1/verify_candidate2.py",
    "validation_defect": "ml/ocr/project_numeric_v1/CANDIDATE_2_VALIDATION_DEFECT.json",
    "preregistration": "ml/ocr/project_numeric_v1/CANDIDATE_2_PREREGISTRATION.md",
    "tests": "ml/ocr/project_numeric_v1/tests/test_candidate2.py"
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
        raise ProtocolViolation("Candidate 2 source binding version is invalid.")
    sources = binding.get("sources")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_SOURCE_ROLES):
        raise ProtocolViolation("Candidate 2 source binding roles are incomplete.")
    for role, relative in REQUIRED_SOURCE_ROLES.items():
        record = sources[role]
        path = root / relative
        if record.get("path") != relative:
            raise ProtocolViolation(f"Candidate 2 source binding path mismatch for {role}.")
        if not path.is_file():
            raise ProtocolViolation(f"Candidate 2 bound source is missing for {role}.")
        if record.get("bytes") != path.stat().st_size:
            raise ProtocolViolation(f"Candidate 2 source byte count mismatch for {role}.")
        if record.get("sha256") != _sha256(path):
            raise ProtocolViolation(f"Candidate 2 source SHA-256 mismatch for {role}.")
    measured = {
        split: split_fingerprint(build_candidate2_split(split))
        for split in ("train", "validation", "sealed_test")
    }
    if measured != frozen["split_fingerprints"]:
        raise ProtocolViolation("Candidate 2 procedural split fingerprints do not match.")
    if binding.get("split_fingerprints") != measured:
        raise ProtocolViolation("Candidate 2 binding split fingerprints do not match.")
    return {
        "binding_valid": True,
        "source_count": len(sources),
        "split_fingerprints": measured,
    }


def verify_committed_preregistration(root: Path = REPOSITORY_ROOT) -> str:
    branch = _run_git(["branch", "--show-current"], root)
    if branch.returncode != 0 or branch.stdout.strip() != "main":
        raise ProtocolViolation("Candidate 2 training requires committed main.")
    head = _run_git(["rev-parse", "HEAD"], root)
    if head.returncode != 0 or len(head.stdout.strip()) != 40:
        raise ProtocolViolation("Candidate 2 requires a resolvable preregistration commit.")
    committed_paths = [
        *REQUIRED_SOURCE_ROLES.values(),
        "ml/ocr/project_numeric_v1/SOURCE_BINDING_CANDIDATE_2.json",
    ]
    for relative in committed_paths:
        exists = _run_git(["cat-file", "-e", f"HEAD:{relative}"], root)
        if exists.returncode != 0:
            raise ProtocolViolation(f"Candidate 2 source is not committed at HEAD: {relative}")
    status = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *committed_paths],
        root,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ProtocolViolation("Candidate 2 requires an unchanged committed source bundle.")
    verification = verify_source_binding(root)
    if verification["binding_valid"] is not True:
        raise ProtocolViolation("Candidate 2 source binding did not verify.")
    return head.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-committed", action="store_true")
    arguments = parser.parse_args()
    result: dict[str, object] = verify_source_binding()
    if arguments.require_committed:
        result["committed_head"] = verify_committed_preregistration()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
