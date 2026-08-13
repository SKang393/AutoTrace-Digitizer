# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for a selected V10 candidate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import acquire_gate_seal, canonical_json_bytes, complete_gate_seal, sha256_file
from .dataset import encode_proposal, load_sealed_public_archive, proposal_summary, proposals, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, THRESHOLDS, TRUTH_MATCH_IOU_MINIMUM
from ml.ocr.component_context_detector_v7.dataset import box_iou


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/component_spaced_recall_detector_v10")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "sealed_gate.py",
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"), Path("ml/markers/gate_seal.py"),
)
GATE_CONFIG = {
    "evaluation_limit": 1, "exact_region_count_every_fixture": True, "false_region_count": 0,
    "missed_region_count": 0, "duplicate_region_count": 0, "prohibited_structure_hits": 0,
    "provider": "CPUExecutionProvider", "direct_fixture_byte_execution_required": True,
}


def evaluate_scenes(scenes: tuple[object, ...], runner: object, threshold: float) -> dict[str, object]:
    cases, true_positives, false_positives, false_negatives, duplicates = [], 0, 0, 0, 0
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = np.asarray(runner(values), dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        probabilities = exponent[:, 1] / exponent.sum(axis=1)
        accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= threshold]
        matched: set[int] = set()
        scene_fp = scene_dup = 0
        for candidate in accepted:
            matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches:
                scene_fp += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index]))
            if best in matched:
                scene_dup += 1
            else:
                matched.add(best)
        scene_fn = len(scene.truths) - len(matched)
        true_positives += len(matched); false_positives += scene_fp; false_negatives += scene_fn; duplicates += scene_dup
        cases.append({
            "scene_id": scene.scene_id, "true_positives": len(matched), "false_positives": scene_fp,
            "false_negatives": scene_fn, "duplicate_region_count": scene_dup,
            "prohibited_structure_hits": scene_fp, "exact": scene_fp == scene_fn == scene_dup == 0,
        })
    return {
        "scene_count": len(scenes), "truth_region_count": sum(len(scene.truths) for scene in scenes),
        "exact_scene_count": sum(int(case["exact"]) for case in cases), "true_positives": true_positives,
        "false_positives": false_positives, "false_negatives": false_negatives,
        "duplicate_region_count": duplicates, "prohibited_structure_hits": false_positives, "cases": cases,
    }


def evaluate_candidate(*, onnx_path: Path, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"OCR V10 public output exists: {output_path}")
    selection = json.loads(selection_report_path.read_text(encoding="utf-8"))
    onnx_sha, selection_sha = sha256_file(onnx_path), sha256_file(selection_report_path)
    threshold = float(selection.get("selected_threshold", -1.0))
    if (
        selection.get("task") != TASK or selection.get("revision") != REVISION
        or selection.get("status") != "selected_public_gate_authorized"
        or selection.get("selection_gate_passed") is not True
        or selection.get("onnx_sha256") != onnx_sha or threshold not in THRESHOLDS
    ):
        raise RuntimeError("Only an exact authorized V10 selection may open public gate")
    config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("OCR V10 public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    archive_path, manifest_path = REPO_ROOT / seal["fixture_archive_path"], REPO_ROOT / seal["private_manifest_path"]
    if sha256_file(archive_path) != seal["fixture_archive_sha256"] or sha256_file(manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("OCR V10 public fixture bytes changed")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V10 public gate requires CPU only")
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": onnx_sha, "selection_report_sha256": selection_sha},
        dataset_manifest_sha256=seal["private_manifest_sha256"], split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
    )
    input_digest, output_digest, calls = sha256(), sha256(), 0
    def runner(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        contiguous = np.ascontiguousarray(values, dtype=np.float32)
        input_digest.update(contiguous.tobytes())
        output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
        output_digest.update(np.ascontiguousarray(output).tobytes()); calls += 1
        return output
    started = time.perf_counter()
    scenes = load_sealed_public_archive(archive_path)
    summary = proposal_summary(scenes)
    if any(summary[key] != seal[key] for key in summary) or split_fingerprint(scenes) != seal["split_fingerprint"]:
        raise RuntimeError("OCR V10 public fixtures violate frozen contract")
    metrics = evaluate_scenes(scenes, runner, threshold)
    passed = metrics["exact_scene_count"] == metrics["scene_count"] and metrics["true_positives"] == metrics["truth_region_count"] and metrics["false_positives"] == metrics["false_negatives"] == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0 and calls == len(scenes)
    report: dict[str, object] = {
        "schema": "graphreader.ocr-spaced-component-recall-public-gate.v1", "task": TASK,
        "revision": PUBLIC_REVISION, "status": "pass" if passed else "fail",
        "production_approval": False, "release_eligible": False, "evaluation_count": 1,
        "onnx_sha256": onnx_sha, "selection_report_sha256": selection_sha,
        "fixture_archive_sha256": seal["fixture_archive_sha256"], "provider": "CPUExecutionProvider",
        "selected_threshold": threshold, "metrics": metrics,
        "direct_execution": {"inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(), "output_tensor_stream_sha256": output_digest.hexdigest()},
        "gate_requirements": GATE_CONFIG, "seal_binding": gate.binding, "canonical_seal_key": gate.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=str(report["status"]), report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_candidate(onnx_path=REPO_ROOT / args.onnx, selection_report_path=REPO_ROOT / args.selection_report, output_path=REPO_ROOT / args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
