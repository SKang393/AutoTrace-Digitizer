# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministically re-score recorded OCR aggregates against the product bar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "ml/policy/ocr-product-bar-v1.json"
RESULT_PATH = REPO_ROOT / "ml/policy/ocr-product-bar-v1-result.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score(candidate: dict[str, Any], bar: dict[str, Any]) -> dict[str, Any]:
    result_path = REPO_ROOT / candidate["result_path"]
    checkpoint_path = REPO_ROOT / candidate["checkpoint_path"]
    onnx_path = REPO_ROOT / candidate["onnx_path"]
    for path, expected in (
        (result_path, candidate["result_sha256"]),
        (checkpoint_path, candidate["checkpoint_sha256"]),
        (onnx_path, candidate["onnx_sha256"]),
    ):
        if _sha256(path) != expected:
            raise RuntimeError(f"OCR product-bar evidence checksum mismatch: {path}")

    recorded = _read_json(result_path)
    if (recorded.get("revision"), recorded.get("candidate_id")) != (
        candidate["revision"], candidate["candidate_id"],
    ):
        raise RuntimeError("OCR product-bar candidate identity mismatch")
    if recorded.get(candidate["checkpoint_hash_field"]) != candidate["checkpoint_sha256"]:
        raise RuntimeError("OCR product-bar checkpoint identity mismatch")
    if recorded.get(candidate["onnx_hash_field"]) != candidate["onnx_sha256"]:
        raise RuntimeError("OCR product-bar ONNX identity mismatch")

    metrics: dict[str, Any] = recorded
    for key in candidate["metrics_path"]:
        metrics = metrics[key]
    scene_count = int(metrics["scene_count"] if "scene_count" in metrics else recorded["scene_count"])
    exact_scene_count = int(metrics["exact_scene_count"])
    scene_exact_rate = exact_scene_count / scene_count
    selected_threshold = float(recorded["selected_threshold"])
    threshold_row = next(
        item for item in recorded["threshold_comparisons"]
        if float(item["threshold"]) == selected_threshold
    )
    if int(threshold_row["prohibited_structure_hits"]) != int(metrics["prohibited_structure_hits"]):
        raise RuntimeError("Selected-threshold prohibited-hit metrics disagree")
    minimum_per_role_accuracy = float(
        metrics["minimum_per_role_accuracy"]
        if "minimum_per_role_accuracy" in metrics
        else threshold_row["minimum_per_role_accuracy"]
    )

    gates = {
        "scene_exact_rate": scene_exact_rate >= bar["scene_exact_rate_minimum"]["value"],
        "character_error_rate": (
            float(metrics["character_error_rate"])
            <= bar["character_error_rate_maximum"]["value"]
        ),
        "role_accuracy": float(metrics["role_accuracy"]) >= bar["role_accuracy_minimum"]["value"],
        "prohibited_structure_hits": (
            int(metrics["prohibited_structure_hits"])
            <= bar["prohibited_structure_hits_maximum"]["value"]
        ),
    }
    passed = all(gates.values())
    sealed = candidate["evidence_split"] == "sealed"
    return {
        "name": candidate["name"],
        "revision": candidate["revision"],
        "candidate_id": candidate["candidate_id"],
        "evidence_split": candidate["evidence_split"],
        "result_path": candidate["result_path"],
        "result_sha256": candidate["result_sha256"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "onnx_sha256": candidate["onnx_sha256"],
        "selected_threshold": selected_threshold,
        "metrics": {
            "scene_count": scene_count,
            "exact_scene_count": exact_scene_count,
            "scene_exact_rate": scene_exact_rate,
            "recognition_exact": float(metrics["recognition_exact"]),
            "character_error_rate": float(metrics["character_error_rate"]),
            "role_accuracy": float(metrics["role_accuracy"]),
            "minimum_per_role_accuracy": minimum_per_role_accuracy,
            "prohibited_structure_hits": int(metrics["prohibited_structure_hits"]),
        },
        "gates": gates,
        "product_bar_passed": passed,
        "qualification": (
            "synthetic_candidate_approved_private_acceptance_pending"
            if passed and sealed
            else "dev_product_bar_pass_supporting_evidence_only"
            if passed
            else "product_bar_failed"
        ),
    }


def rescore() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    policy_path = REPO_ROOT / protocol["evidence_policy"]["path"]
    if _sha256(policy_path) != protocol["evidence_policy"]["sha256"]:
        raise RuntimeError("Shared evidence-policy checksum mismatch")
    candidates = [_score(candidate, protocol["acceptance_bar"]) for candidate in protocol["candidates"]]
    approved = next(
        (item for item in candidates if item["product_bar_passed"] and item["evidence_split"] == "sealed"),
        None,
    )
    return {
        "schema_version": 1,
        "task": protocol["task"],
        "policy_path": PROTOCOL_PATH.relative_to(REPO_ROOT).as_posix(),
        "policy_sha256": _sha256(PROTOCOL_PATH),
        "evaluation_mode": "recorded_aggregate_metrics_only",
        "model_training_runs": 0,
        "model_inference_runs": 0,
        "sealed_split_reads": 0,
        "case_level_reads": 0,
        "candidates": candidates,
        "status": "pass" if approved is not None else "fail",
        "selected_candidate": None if approved is None else {
            "revision": approved["revision"],
            "candidate_id": approved["candidate_id"],
            "checkpoint_sha256": approved["checkpoint_sha256"],
            "onnx_sha256": approved["onnx_sha256"],
        },
        "synthetic_candidate_approval": approved is not None,
        "private_acceptance": False,
        "production_approval": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "packaging_discovery": False,
        "release_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generated = rescore()
    if arguments.check and _read_json(RESULT_PATH) != generated:
        raise RuntimeError("Tracked OCR product-bar result does not match recorded aggregate evidence")
    print(json.dumps(generated, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
