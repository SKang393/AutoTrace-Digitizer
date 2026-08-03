# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import jsonschema


WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".engine",
    ".onnx",
    ".param",
    ".pdparams",
    ".plan",
    ".pt",
    ".pth",
    ".safetensors",
}


def test_graphsr_manifests_obey_frozen_schema_and_provenance(
    repository_root: Path,
) -> None:
    schema = json.loads(
        (repository_root / "contracts" / "model-manifest.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest_root = repository_root / "models" / "manifest" / "graphsr"
    manifests = sorted(manifest_root.glob("*.json"))
    assert manifests, "Session 07 must publish at least one metadata-only manifest"

    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(manifest)
        assert manifest["task"] == "super_resolution"
        assert manifest["license"]["reviewed"] is True
        assert manifest["license"]["spdx"]
        notice = repository_root / manifest["license"]["notice_path"]
        assert notice.is_file(), f"Missing reviewed notice: {notice}"
        assert manifest["commercial_use"] is True
        assert isinstance(manifest["redistribution"], bool)
        assert set(manifest["providers"]) <= {
            "cpu",
            "directml",
            "winml",
            "cuda",
            "openvino",
            "vulkan",
        }
        assert manifest["sha256"] != "0" * 64, "Artifact digests cannot be placeholders"
        assert any(name.lower().endswith(".onnx") for name in manifest["files"])
        assert manifest["inputs"] and manifest["outputs"]


def test_unmeasured_manifest_benchmarks_do_not_fabricate_metrics(
    repository_root: Path,
) -> None:
    manifest_root = repository_root / "models" / "manifest" / "graphsr"
    for path in manifest_root.glob("*.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for benchmark in manifest.get("benchmarks", []):
            if benchmark.get("status") in {"not_run", "blocked", "unmeasured"}:
                values = benchmark.get("metrics")
                assert values is None or all(value is None for value in values.values())
                assert benchmark.get("selected_as_default") is not True


def test_model_weights_and_generated_training_data_are_not_tracked(
    repository_root: Path,
) -> None:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    tracked = [Path(raw.decode("utf-8")) for raw in result.stdout.split(b"\0") if raw]
    forbidden = [
        path
        for path in tracked
        if path.suffix.lower() in WEIGHT_SUFFIXES
        or "ml/graphsr/checkpoints/" in path.as_posix().lower()
        or "ml/graphsr/datasets/" in path.as_posix().lower()
    ]
    assert forbidden == []


def test_dependency_ledger_notice_hashes_and_requirements_are_complete(
    repository_root: Path,
) -> None:
    graphsr_root = repository_root / "ml" / "graphsr"
    with (graphsr_root / "DEPENDENCY_PROVENANCE.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = tuple(csv.DictReader(stream))
    assert {row["dependency"] for row in rows} == {
        "PyTorch",
        "ONNX",
        "ONNX Runtime",
        "NumPy",
        "Pillow",
        "psutil",
        "pytest",
        "jsonschema",
    }
    for row in rows:
        notice = graphsr_root / row["notice_path"]
        assert notice.is_file()
        assert row["notice_sha256"] == f"sha256:{hashlib.sha256(notice.read_bytes()).hexdigest()}"
        assert row["artifact_checksum"].startswith("sha256:")
        assert len(row["artifact_checksum"]) == 71
        assert row["review_status"].startswith("approved-permissive")

    requirements = "\n".join(
        (
            (graphsr_root / "requirements.txt").read_text(encoding="utf-8"),
            (graphsr_root / "requirements-test.txt").read_text(encoding="utf-8"),
        )
    ).lower()
    for row in rows:
        normalized = {
            "PyTorch": "torch",
            "ONNX Runtime": "onnxruntime",
        }.get(row["dependency"], row["dependency"]).lower()
        assert f"{normalized}=={row['version']}" in requirements
        assert row["artifact_checksum"].removeprefix("sha256:") in requirements
