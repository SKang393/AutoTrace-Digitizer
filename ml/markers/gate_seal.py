# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Repository-scoped, source-bound scientific gate seals."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Mapping, Sequence


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def source_bundle_sha256(repo_root: Path, paths: Sequence[Path]) -> str:
    rows = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        resolved = path if path.is_absolute() else repo_root / path
        relative = resolved.relative_to(repo_root).as_posix()
        rows.append(f"{relative}={sha256_file(resolved)}\n")
    return sha256_bytes("".join(rows).encode("utf-8"))


def require_committed_sources(repo_root: Path, paths: Sequence[Path]) -> None:
    relative = [str((path if path.is_absolute() else repo_root / path).relative_to(repo_root)) for path in paths]
    for path in relative:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", path],
            cwd=repo_root,
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"Evidence source must be committed before use: {path}")
    for args in (("git", "diff", "--quiet", "--", *relative), ("git", "diff", "--cached", "--quiet", "--", *relative)):
        result = subprocess.run(args, cwd=repo_root, check=False)
        if result.returncode != 0:
            raise RuntimeError("Evidence sources or configurations differ from the committed revision")


@dataclass(frozen=True)
class GateSeal:
    key: str
    directory: Path
    opened_path: Path
    binding: dict[str, object]


def require_evaluator_identity(
    *,
    expected_task: str,
    expected_revision: str,
    manifest: Mapping[str, object],
    split_config: Mapping[str, object],
    seal_binding: Mapping[str, object] | None = None,
    report: Mapping[str, object] | None = None,
) -> None:
    payloads: list[tuple[str, Mapping[str, object]]] = [
        ("manifest", manifest),
        ("split", split_config),
    ]
    if seal_binding is not None:
        payloads.append(("seal", seal_binding))
    if report is not None:
        payloads.append(("report", report))
    for name, payload in payloads:
        if payload.get("task") != expected_task:
            raise RuntimeError(f"{name} task does not match frozen gate identity: {payload.get('task')}")
        if payload.get("revision") != expected_revision:
            raise RuntimeError(f"{name} revision does not match frozen gate identity: {payload.get('revision')}")


def acquire_gate_seal(
    *,
    repo_root: Path,
    task: str,
    revision: str,
    candidate_hashes: Mapping[str, str],
    dataset_manifest_sha256: str,
    split_config_path: Path,
    evaluator_source_paths: Sequence[Path],
    gate_config: Mapping[str, object],
) -> GateSeal:
    canonical_root = repo_root / "ml" / "markers" / "gate-seals"
    retired_path = canonical_root / "retired-historical-pairs.json"
    if not retired_path.is_file():
        raise RuntimeError(f"Canonical retired-pair policy is missing: {retired_path}")
    source_paths = tuple(evaluator_source_paths) + (
        split_config_path,
        retired_path.relative_to(repo_root),
    )
    require_committed_sources(repo_root, source_paths)
    split_config = json.loads((repo_root / split_config_path).read_text(encoding="utf-8"))
    expected_task = split_config.get("task")
    if task != expected_task:
        raise RuntimeError(f"Gate task {task} does not match frozen configuration {expected_task}")
    expected_revision = split_config.get("revision")
    if revision != expected_revision:
        raise RuntimeError(f"Gate revision {revision} does not match frozen configuration {expected_revision}")
    expected_candidate_hash_keys = split_config.get("expected_candidate_hash_keys")
    actual_candidate_hash_keys = list(candidate_hashes.keys())
    if actual_candidate_hash_keys != expected_candidate_hash_keys:
        raise RuntimeError(
            "Candidate hash key schema "
            f"{actual_candidate_hash_keys} does not match frozen configuration {expected_candidate_hash_keys}"
        )
    expected_manifest = split_config.get("expected_dataset_manifest_sha256")
    if expected_manifest != dataset_manifest_sha256:
        raise RuntimeError(
            f"Generated split manifest {dataset_manifest_sha256} does not match frozen configuration {expected_manifest}"
        )
    evaluator_sha256 = source_bundle_sha256(repo_root, evaluator_source_paths)
    expected_evaluator = split_config.get("expected_evaluator_source_bundle_sha256")
    if expected_evaluator != evaluator_sha256:
        raise RuntimeError(
            f"Evaluator source bundle {evaluator_sha256} does not match frozen configuration {expected_evaluator}"
        )
    split_config_sha256 = sha256_file(repo_root / split_config_path)
    gate_config_sha256 = sha256_bytes(canonical_json_bytes(dict(gate_config)))
    expected_gate_config = split_config.get("expected_gate_config_sha256")
    if expected_gate_config != gate_config_sha256:
        raise RuntimeError(
            f"Runtime gate configuration {gate_config_sha256} does not match frozen configuration {expected_gate_config}"
        )
    binding: dict[str, object] = {
        "task": task,
        "revision": revision,
        "candidate_hashes": dict(sorted(candidate_hashes.items())),
        "candidate_hash_key_schema": actual_candidate_hash_keys,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "split_config_path": split_config_path.as_posix(),
        "split_config_sha256": split_config_sha256,
        "evaluator_source_paths": sorted(path.as_posix() for path in evaluator_source_paths),
        "evaluator_source_bundle_sha256": evaluator_sha256,
        "gate_config_sha256": gate_config_sha256,
        "ledger_mode": "canonical_repository",
        "ledger_root": "ml/markers/gate-seals",
        "committed_source_enforcement": True,
        "retired_policy_sha256": sha256_file(retired_path),
    }
    replay_identity = {
        "task": task,
        "revision": revision,
        "candidate_hashes": dict(sorted(candidate_hashes.items())),
    }
    key = sha256_bytes(canonical_json_bytes(replay_identity))
    retired = json.loads(retired_path.read_text(encoding="utf-8"))
    if any(
        item.get("key") == key
        or (
            item.get("task") == task
            and item.get("revision") == revision
            and item.get("candidate_hashes") == dict(sorted(candidate_hashes.items()))
        )
        for item in retired.get("pairs", [])
    ):
        raise RuntimeError(f"Gate pair is retired historical evidence and cannot be replayed: {key}")
    directory = canonical_root / task / key
    directory.mkdir(parents=True, exist_ok=True)
    opened_path = directory / "opened.json"
    opened = {
        "schema_version": 1,
        "status": "opened",
        "evaluation_count": 1,
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "key": key,
        "binding": binding,
    }
    try:
        with opened_path.open("xb") as stream:
            stream.write(canonical_json_bytes(opened))
    except FileExistsError as error:
        raise RuntimeError(f"Gate candidate/revision pair was already opened: {key}") from error
    return GateSeal(key, directory, opened_path, binding)


def complete_gate_seal(seal: GateSeal, *, status: str, report_sha256: str) -> Path:
    result_path = seal.directory / "result.json"
    result = {
        "schema_version": 1,
        "status": status,
        "evaluation_count": 1,
        "key": seal.key,
        "opened_sha256": sha256_file(seal.opened_path),
        "report_sha256": report_sha256,
    }
    try:
        with result_path.open("xb") as stream:
            stream.write(canonical_json_bytes(result))
    except FileExistsError as error:
        raise RuntimeError(f"Gate result was already recorded: {seal.key}") from error
    return result_path


__all__ = [
    "GateSeal",
    "acquire_gate_seal",
    "canonical_json_bytes",
    "complete_gate_seal",
    "require_evaluator_identity",
    "require_committed_sources",
    "sha256_bytes",
    "sha256_file",
    "source_bundle_sha256",
]
