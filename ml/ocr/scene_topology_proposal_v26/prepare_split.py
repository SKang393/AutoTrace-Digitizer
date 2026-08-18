# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-time fixture freeze for OCR scene-topology proposal V26."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
from typing import Any

from PIL import Image

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256

from .dataset import build_split, save_archive
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    ROLE_PARENT_CHECKPOINT_PATH,
    ROLE_PARENT_CHECKPOINT_SHA256,
    ROLE_PARENT_ONNX_PATH,
    ROLE_PARENT_ONNX_SHA256,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/scene_topology_proposal_v26")
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
ARCHIVE_PATHS = {
    "train": Path("artifacts/production-validation/ocr-v26-train.zip"),
    "validation": Path("artifacts/production-validation/ocr-v26-selection.zip"),
    "sealed_public": Path("artifacts/production-validation/ocr-v26-public.zip"),
}
SOURCE_PATHS = (
    ROOT / "PROTOCOL.json",
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "prepare_split.py",
    ROOT / "protocol.py",
    ROOT / "sealed_gate.py",
    ROOT / "train_p1.py",
    Path(TRIGGER_RESULT_PATH),
    Path("ml/ocr/evidence_rescue_v25/model.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/dataset.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/model_p2.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/pipeline.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/protocol.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/train_p1.py"),
    Path("ml/ocr/role_anchor_set_v23/dataset.py"),
    Path("ml/ocr/role_anchor_set_v23/model.py"),
    Path("ml/ocr/role_anchor_set_v23/protocol.py"),
    Path("ml/ocr/margin_calibrator_v20/dataset.py"),
    Path("ml/ocr/margin_calibrator_v20/pipeline.py"),
    Path("ml/ocr/margin_calibrator_v20/protocol.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/dataset.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/protocol.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)


def _repository_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _require_sources_at_head() -> None:
    for path in SOURCE_PATHS:
        relative = path.as_posix()
        local = REPO_ROOT / path
        if not local.is_file():
            raise RuntimeError(f"OCR V26 required source is missing: {relative}")
        completed = subprocess.run(
            ("git", "show", f"HEAD:{relative}"), cwd=REPO_ROOT,
            check=False, capture_output=True,
        )
        if completed.returncode != 0 or completed.stdout != local.read_bytes():
            raise RuntimeError(f"OCR V26 source must be committed unchanged before freeze: {relative}")
    expected = {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        ROLE_PARENT_CHECKPOINT_PATH: ROLE_PARENT_CHECKPOINT_SHA256,
        ROLE_PARENT_ONNX_PATH: ROLE_PARENT_ONNX_SHA256,
        TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
    }
    for relative, digest in expected.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"OCR V26 exact prerequisite changed: {relative}")


def _source_hash(scene: Any) -> str:
    stream = BytesIO()
    Image.fromarray(scene.raster, mode="L").save(
        stream, format="PNG", optimize=False, compress_level=9,
    )
    return sha256(stream.getvalue()).hexdigest()


def freeze() -> dict[str, object]:
    if (REPO_ROOT / SEAL_PATH).exists():
        raise RuntimeError("OCR V26 split identity is already frozen")
    existing = [path.as_posix() for path in ARCHIVE_PATHS.values() if (REPO_ROOT / path).exists()]
    if existing:
        raise RuntimeError("OCR V26 archive already exists before freeze: " + ", ".join(existing))
    _require_sources_at_head()
    records: dict[str, object] = {}
    hash_sets: dict[str, set[str]] = {}
    for split, relative in ARCHIVE_PATHS.items():
        scenes = build_split(split)  # type: ignore[arg-type]
        hashes = {_source_hash(scene) for scene in scenes}
        if len(hashes) != len(scenes):
            raise RuntimeError(f"OCR V26 {split} contains duplicate fixture bytes")
        hash_sets[split] = hashes
        records[split] = {
            **save_archive(scenes, REPO_ROOT / relative),
            "source_sha256_inventory": sorted(hashes),
        }
    overlaps = {
        "train_validation": len(hash_sets["train"] & hash_sets["validation"]),
        "train_sealed_public": len(hash_sets["train"] & hash_sets["sealed_public"]),
        "validation_sealed_public": len(hash_sets["validation"] & hash_sets["sealed_public"]),
    }
    if any(overlaps.values()):
        raise RuntimeError("OCR V26 split fixture bytes overlap")
    value: dict[str, object] = {
        "schema": "graphreader.ocr-scene-topology-split-seal.v1",
        "revision": REVISION,
        "source_commit": _repository_head(),
        "protocol": protocol_configuration(),
        "source_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS
        },
        "source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SOURCE_PATHS),
        "splits": records,
        "cross_split_source_overlap_counts": overlaps,
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }
    (REPO_ROOT / SEAL_PATH).write_bytes(canonical_json_bytes(value))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze",))
    parser.parse_args()
    result = freeze()
    print(json.dumps({
        "source_commit": result["source_commit"],
        "source_bundle_sha256": result["source_bundle_sha256"],
        "cross_split_source_overlap_counts": result["cross_split_source_overlap_counts"],
        "splits": {
            name: {
                "archive_sha256": record["archive_sha256"],
                "split_fingerprint": record["split_fingerprint"],
                "proposal_summary": record["proposal_summary"],
            }
            for name, record in result["splits"].items()
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ARCHIVE_PATHS", "SEAL_PATH", "SOURCE_PATHS", "freeze"]
