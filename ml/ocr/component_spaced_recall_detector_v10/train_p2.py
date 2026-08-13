# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V10 P2 fine-tune for spaced text recall."""

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

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.component_fusion_detector_v8.train_p1 import _balanced_order, _configure, _export
from ml.ocr.component_recall_detector_v9.model import ComponentRecallNet
from ml.ocr.component_recall_detector_v9.model_p3 import ScaledComponentRecallNet

from .dataset import build_split, encode_proposal, proposal_summary, proposals, split_fingerprint
from .pipeline import evaluate_thresholds
from .protocol import REVISION, TASK, THRESHOLDS
from .training_data_p2 import training_examples


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/component_spaced_recall_detector_v10")
CONFIG_PATH = ROOT / "training/p2.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p2.py",
    ROOT / "training_data_p2.py",
    Path("ml/ocr/component_recall_detector_v9/dataset.py"),
    Path("ml/ocr/component_recall_detector_v9/model.py"),
    Path("ml/ocr/component_recall_detector_v9/model_p3.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline.py"),
    Path("ml/ocr/component_recall_detector_v9/pipeline_p2.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_fusion_detector_v8/protocol.py"),
    Path("ml/ocr/component_fusion_detector_v8/train_p1.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/production_composition_v1/protocol.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _selection_values() -> tuple[object, ...]:
    return build_split("validation")


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V10 P2 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id="P2",
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase, optimizer_steps = "initialization", 0
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V10 P2 runner source changed")
        if config["p1_result_sha256"] != sha256_file(REPO_ROOT / P1_RESULT_PATH):
            raise RuntimeError("OCR V10 P1 evidence changed")
        p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
        if p1.get("status") != "failed_selection" or p1.get("selection_false_negatives") != 4 or p1.get("public_gate_archive_opened") is not False:
            raise RuntimeError("OCR V10 P1 evidence does not authorize P2")
        if (
            config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH)
            or config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH)
        ):
            raise RuntimeError("OCR V10 split evidence changed")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        validation = _selection_values()
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"] or proposal_summary(validation) != {key: selection["validation"][key] for key in proposal_summary(validation)}:
            raise RuntimeError("OCR V10 validation generator changed")
        values, labels, training_evidence = training_examples()
        if any(
            training_evidence[key] != config[key]
            for key in (
                "scene_count",
                "negative_cap_per_scene",
                "proposal_count",
                "positive_proposal_count",
                "negative_proposal_count",
                "tensor_label_stream_sha256",
            )
        ) or training_evidence["validation_or_public_pixels_used"] is not False:
            raise RuntimeError("OCR V10 P2 training examples changed")
        source_checkpoint = REPO_ROOT / config["source_checkpoint_path"]
        if sha256_file(source_checkpoint) != config["source_checkpoint_sha256"]:
            raise RuntimeError("OCR V10 P2 source checkpoint changed")
        base = ComponentRecallNet(seed=int(config["seed"]))
        checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        base.load_state_dict(checkpoint["state_dict"])
        base.train()
        generator = _configure(int(config["seed"]))
        optimizer = torch.optim.AdamW(base.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        criterion = nn.CrossEntropyLoss()
        tensors, targets = torch.from_numpy(values), torch.from_numpy(labels)
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        for epoch in range(int(config["epochs"])):
            order = _balanced_order(targets, generator)
            losses: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                loss = criterion(base(tensors.index_select(0, indices)), targets.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) - 1}:
                checkpoints.append({"epoch": epoch + 1, "balanced_cross_entropy": sum(losses) / len(losses)})
        phase = "export"
        base.eval()
        model = ScaledComponentRecallNet(base).eval()
        checkpoint_path = output_dir / "graph-text-spaced-component-recall-v10-p2.pt"
        torch.save({"state_dict": base.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-spaced-component-recall-v10-p2.onnx"
        parity_values = torch.from_numpy(values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        input_digest = __import__("hashlib").sha256()
        output_digest = __import__("hashlib").sha256()
        calls = 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            contiguous = np.ascontiguousarray(input_values, dtype=np.float32)
            input_digest.update(contiguous.tobytes())
            output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
            output_digest.update(np.ascontiguousarray(output).tobytes())
            calls += 1
            return output

        phase = "selection"
        comparisons = evaluate_thresholds(validation, runner, THRESHOLDS)
        selected = max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], -item["metrics"]["duplicate_region_count"], item["threshold"],
        ))
        metrics = selected["metrics"]
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["true_positives"] == metrics["truth_region_count"]
            and metrics["false_positives"]
            == metrics["false_negatives"]
            == metrics["duplicate_region_count"]
            == metrics["prohibited_structure_hits"]
            == 0
            and parity_passed
            and calls == len(validation)
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-spaced-component-recall-candidate.v1", "task": TASK,
            "revision": REVISION, "candidate_id": "P2", "status": "selected" if passed else "failed_selection",
            "production_approval": False, "release_eligible": False,
            "synthetic_only": True, "private_or_article_images": False,
            "chandler_included": False, "generalization_label_included": False,
            "isolated_change": config["isolated_change"],
            "optimizer_steps": optimizer_steps, "epochs": config["epochs"], "training_evidence": training_evidence,
            "loss_checkpoints": checkpoints, "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": maximum_error, "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider", "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"], "selection_metrics": metrics,
            "selection_gate_passed": passed, "direct_execution": {
                "inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(),
                "output_tensor_stream_sha256": output_digest.hexdigest(),
            },
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-spaced-component-recall-failure.v1", "task": TASK,
            "revision": REVISION, "candidate_id": "P2", "status": "failed_runner",
            "phase": phase, "optimizer_steps": optimizer_steps, "exception_type": type(error).__name__,
            "exception_message": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False, "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False, "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "optimizer_steps": report["optimizer_steps"], "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"], "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
