# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P2 hard-negative repair training for OCR detector V9."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.component_fusion_detector_v8.train_p1 import _balanced_order, _configure, _export, _runner

from .dataset import build_split, proposal_examples, split_fingerprint
from .dataset_p2 import p2_proposal_examples
from .model import ComponentRecallNet
from .pipeline_p2 import evaluate_thresholds
from .protocol import REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/ocr/component_recall_detector_v9/training/p2.json")
CANONICAL_OUTPUT = Path("ml/ocr/component_recall_detector_v9/artifacts/P2-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/dataset_p2.py"),
    Path("ml/ocr/component_recall_detector_v9/model.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline_p2.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_recall_detector_v9/train_p2.py"),
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_fusion_detector_v8/train_p1.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    optimizer_steps = 0
    try:
        p1_result_path = REPO_ROOT / config["p1_result_path"]
        if sha256_file(p1_result_path) != config["p1_result_sha256"]:
            raise RuntimeError("OCR V9 P1 result checksum changed")
        p1_result = json.loads(p1_result_path.read_text(encoding="utf-8"))
        p1_report_path = REPO_ROOT / p1_result["candidate_report_path"]
        if sha256_file(p1_report_path) != p1_result["candidate_report_sha256"]:
            raise RuntimeError("OCR V9 P1 direct report checksum changed")
        if (
            p1_result.get("status") != "failed_selection"
            or p1_result.get("selection_false_positives") != 11
            or p1_result.get("selection_false_negatives") != 0
            or p1_result.get("public_gate_archive_opened") is not False
        ):
            raise RuntimeError("OCR V9 P1 evidence does not authorize the P2 defect repair")
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("OCR V9 selection manifest checksum changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("OCR V9 public seal checksum changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / seal["fixture_archive_path"]) != seal["fixture_archive_sha256"]:
            raise RuntimeError("OCR V9 sealed-public archive changed before P2 training")
        seed = int(config["seed"])
        generator = _configure(seed)
        model = ComponentRecallNet(seed=seed).eval()
        phase = "onnx_preflight"
        preflight_path = output_dir / "export-preflight.onnx"
        _export(model, torch.zeros((8, 2, 32, 140), dtype=torch.float32), preflight_path)
        preflight_sha256 = sha256_file(preflight_path)
        preflight_session = ort.InferenceSession(str(preflight_path), providers=["CPUExecutionProvider"])
        preflight_output = np.asarray(
            preflight_session.run(None, {"region_proposals": np.zeros((3, 2, 32, 140), dtype=np.float32)})[0],
            dtype=np.float32,
        )
        if preflight_output.shape != (3, 2) or not np.isfinite(preflight_output).all():
            raise RuntimeError("OCR V9 P2 export preflight returned an invalid tensor")
        preflight_path.unlink()
        training_scenes = build_split("train")
        validation_scenes = build_split("validation")
        if split_fingerprint(training_scenes) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V9 training renderer changed after freeze")
        if split_fingerprint(validation_scenes) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V9 validation renderer changed after freeze")
        training_values, training_labels, augmentation = p2_proposal_examples(training_scenes)
        if (
            augmentation["augmented_negative_count"] != config["augmented_negative_count"]
            or augmentation["augmented_tensor_stream_sha256"]
            != config["augmented_negative_tensor_stream_sha256"]
            or augmentation["negative_cap_per_scene"] != config["augmented_negative_cap_per_scene"]
            or len(training_labels) != config["training_proposal_count"]
            or int(training_labels.sum()) != config["training_positive_proposal_count"]
            or int(len(training_labels) - training_labels.sum()) != config["training_negative_proposal_count"]
            or augmentation["truth_overlap_allowed"] is not False
        ):
            raise RuntimeError("OCR V9 P2 training-only augmentation changed after preregistration")
        validation_values, _ = proposal_examples(validation_scenes)
        model.train()
        phase = "training"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
        )
        criterion = nn.CrossEntropyLoss()
        values = torch.from_numpy(training_values)
        labels = torch.from_numpy(training_labels)
        batch_size = int(config["batch_size"])
        checkpoints: list[dict[str, float | int]] = []
        for epoch in range(int(config["epochs"])):
            order = _balanced_order(labels, generator)
            losses: list[float] = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                loss = criterion(model(values.index_select(0, indices)), labels.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({"epoch": epoch + 1, "balanced_cross_entropy": sum(losses) / len(losses)})
        phase = "selection"
        model.eval()
        comparisons = evaluate_thresholds(
            validation_scenes, _runner(model), tuple(float(value) for value in config["selection_thresholds"])
        )
        selected = max(
            comparisons,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                -item["metrics"]["false_negatives"],
                -item["metrics"]["duplicate_region_count"],
                item["threshold"],
            ),
        )
        metrics = selected["metrics"]
        selection_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == metrics["false_negatives"] == 0
            and metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
        )
        phase = "export"
        checkpoint_path = output_dir / "graph-text-component-recall-v9-p2.pt"
        torch.save({"state_dict": model.state_dict(), "selected_threshold": selected["threshold"]}, checkpoint_path)
        onnx_path = output_dir / "graph-text-component-recall-v9-p2.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        report: dict[str, object] = {
            "schema": "graphreader.ocr-component-recall-training-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "generalization_label_included": False,
            "predecessor_fixture_bytes_reused": False,
            "prior_validation_pixels_used_for_training": False,
            "training_authorization": authorization.binding,
            "p1_result_path": config["p1_result_path"],
            "p1_result_sha256": config["p1_result_sha256"],
            "isolated_change": config["isolated_change"],
            "training_augmentation_evidence": augmentation,
            "training_scene_count": len(training_scenes),
            "training_proposal_count": len(training_labels),
            "training_positive_proposal_count": int(training_labels.sum()),
            "training_negative_proposal_count": int(len(training_labels) - training_labels.sum()),
            "validation_scene_count": len(validation_scenes),
            "epochs": config["epochs"],
            "seed": seed,
            "optimizer_steps": optimizer_steps,
            "loss_checkpoints": checkpoints,
            "onnx_preflight_sha256": preflight_sha256,
            "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"],
            "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-component-recall-training-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = train_candidate(REPO_ROOT / arguments.output)
    print(json.dumps({
        "candidate_id": report["candidate_id"],
        "status": report["status"],
        "selected_threshold": report["selected_threshold"],
        "selection_gate_passed": report["selection_gate_passed"],
        "onnx_parity_passed": report["onnx_parity_passed"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
