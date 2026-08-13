# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Run the single zero-optimizer V10 P1 threshold selection."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, encode_proposal, proposal_summary, proposals, split_fingerprint
from .protocol import BASE_ONNX_SHA256, REVISION, TASK, THRESHOLDS, TRUTH_MATCH_IOU_MINIMUM
from .sealed_gate import evaluate_scenes


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/component_spaced_recall_detector_v10")
CONFIG_PATH = ROOT / "training/p1.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "evaluate_p1.py", ROOT / "pipeline.py", ROOT / "protocol.py",
    ROOT / "sealed_gate.py", Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"), Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def evaluate_candidate(*, onnx_path: Path, output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V10 P1 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id="P1",
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    if config.get("optimizer_steps") != 0 or config.get("weights_changed") is not False:
        raise RuntimeError("OCR V10 P1 must retain exact weights")
    if sha256_file(onnx_path) != BASE_ONNX_SHA256:
        raise RuntimeError("OCR V10 P1 base ONNX changed")
    if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
        raise RuntimeError("OCR V10 P1 runner source changed")
    if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
        raise RuntimeError("OCR V10 selection manifest changed")
    if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
        raise RuntimeError("OCR V10 public seal changed")
    selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
    validation = build_split("validation")
    if split_fingerprint(validation) != selection["validation"]["split_fingerprint"] or proposal_summary(validation) != {key: selection["validation"][key] for key in proposal_summary(validation)}:
        raise RuntimeError("OCR V10 validation fixtures changed")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V10 P1 requires CPU only")
    input_digest, output_digest, calls = sha256(), sha256(), 0
    def runner(values: np.ndarray) -> np.ndarray:
        nonlocal calls
        contiguous = np.ascontiguousarray(values, dtype=np.float32)
        input_digest.update(contiguous.tobytes())
        output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
        output_digest.update(np.ascontiguousarray(output).tobytes()); calls += 1
        return output
    cached: list[tuple[object, tuple[object, ...], np.ndarray]] = []
    for scene in validation:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        logits = runner(values)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        cached.append((scene, candidates, exponent[:, 1] / exponent.sum(axis=1)))
    comparisons = []
    for threshold in THRESHOLDS:
        def cached_runner(values: np.ndarray) -> np.ndarray:
            raise AssertionError("cached runner must not execute")
        # Compute with the same fixed matching contract without another inference.
        cases, tp, fp, fn, dup = [], 0, 0, 0, 0
        from ml.ocr.component_context_detector_v7.dataset import box_iou
        for scene, candidates, probabilities in cached:
            accepted = [item for item, probability in zip(candidates, probabilities, strict=True) if probability >= threshold]
            matched: set[int] = set(); scene_fp = scene_dup = 0
            for candidate in accepted:
                matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth) >= TRUTH_MATCH_IOU_MINIMUM]
                if not matches: scene_fp += 1; continue
                best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index]))
                if best in matched: scene_dup += 1
                else: matched.add(best)
            scene_fn = len(scene.truths) - len(matched)
            tp += len(matched); fp += scene_fp; fn += scene_fn; dup += scene_dup
            cases.append({"scene_id": scene.scene_id, "exact": scene_fp == scene_fn == scene_dup == 0})
        comparisons.append({"threshold": threshold, "metrics": {
            "scene_count": len(validation), "truth_region_count": sum(len(scene.truths) for scene in validation),
            "exact_scene_count": sum(int(case["exact"]) for case in cases), "true_positives": tp,
            "false_positives": fp, "false_negatives": fn, "duplicate_region_count": dup,
            "prohibited_structure_hits": fp,
        }})
    selected = max(comparisons, key=lambda item: (item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"], -item["metrics"]["false_negatives"], -item["metrics"]["duplicate_region_count"], item["threshold"]))
    metrics = selected["metrics"]
    passed = metrics["exact_scene_count"] == metrics["scene_count"] and metrics["true_positives"] == metrics["truth_region_count"] and metrics["false_positives"] == metrics["false_negatives"] == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0 and calls == len(validation)
    report: dict[str, object] = {
        "schema": "graphreader.ocr-spaced-component-recall-candidate.v1", "task": TASK,
        "revision": REVISION, "candidate_id": "P1", "status": "selected" if passed else "failed_selection",
        "optimizer_steps": 0, "weights_changed": False, "onnx_sha256": BASE_ONNX_SHA256,
        "provider": "CPUExecutionProvider", "threshold_comparisons": comparisons,
        "selected_threshold": selected["threshold"], "selection_metrics": metrics,
        "selection_gate_passed": passed, "direct_execution": {
            "inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(),
            "output_tensor_stream_sha256": output_digest.hexdigest(),
        },
        "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
        "production_approval": False, "release_eligible": False,
        "training_authorization": authorization.binding,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = evaluate_candidate(onnx_path=REPO_ROOT / args.onnx, output_dir=REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
