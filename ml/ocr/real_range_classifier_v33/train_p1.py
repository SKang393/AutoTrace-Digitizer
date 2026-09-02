# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Canonical-budget V33 synthetic train/dev runner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate

from .model import RealRangeClassifierV33
from .pipeline import evaluate_scenes
from .protocol import (
    BATCH_SIZE, CANONICAL_OUTPUT, CLASS_WEIGHTS, EPOCHS, LEARNING_RATE,
    FINE_TUNE_RESULT_PATH, FINE_TUNE_RESULT_SHA256, MODEL_LICENSE,
    OFFICIAL_DIAGNOSTIC_PATH, OFFICIAL_DIAGNOSTIC_SHA256,
    ONNX_PARITY_TOLERANCE, REVISION, SEED, SOURCE_MODEL_REVISION,
    TASK, WEIGHT_DECAY,
)
from .training_data import training_examples
from ml.ocr.real_range_classifier_finetune_v32.dataset import build_split, split_fingerprint


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/real_range_classifier_v33")
CONFIG_PATH = ROOT / "training/p1.json"
SOURCE_PATHS = (
    ROOT / "model.py", ROOT / "pipeline.py", ROOT / "protocol.py",
    ROOT / "train_p1.py", ROOT / "training_data.py",
    Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py"),
    Path("ml/ocr/real_range_classifier_finetune_v32/pipeline.py"),
    Path("ml/ocr/real_range_classifier_finetune_v32/protocol.py"),
    Path("ml/ocr/real_range_classifier_finetune_v32/training_data.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/synthetic/dataset.py"), Path("ml/synthetic/renderer.py"),
    Path("ml/synthetic/templates.py"), Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"), Path("ml/policy/evidence_policy.py"),
)


def _export(model: nn.Module, values: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model.eval(), values[:1], path,
        input_names=["region_proposals"], output_names=["region_logits"],
        dynamic_axes={"region_proposals": {0: "proposal_count"}, "region_logits": {0: "proposal_count"}},
        opset_version=17, dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"V33 output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    if config["expected_runner_source_bundle_sha256"] == "ROOT_TO_FILL_SOURCE_BUNDLE_SHA256":
        raise RuntimeError("V33 source bundle hash must be filled after source review")
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id="P1",
        config_path=CONFIG_PATH, runner_source_paths=SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    steps = 0
    try:
        values, labels, evidence = training_examples()
        tensors = torch.from_numpy(values)
        targets = torch.from_numpy(labels)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(SEED)
        model = RealRangeClassifierV33(SEED)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(CLASS_WEIGHTS, dtype=torch.float32))
        order_generator = torch.Generator().manual_seed(SEED)
        phase = "training"
        loss_trace: list[float] = []
        for _ in range(EPOCHS):
            order = torch.randperm(len(targets), generator=order_generator)
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start : start + BATCH_SIZE]
                loss = criterion(model(tensors.index_select(0, indices)), targets.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                steps += 1
                loss_trace.append(float(loss.detach()))
        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-real-range-classifier-v33-p1.pt"
        torch.save({"state_dict": model.state_dict(), "seed": SEED, "architecture": "multiscale-detail-context-geometry-fusion-v1"}, checkpoint_path)
        onnx_path = output_dir / "graph-text-real-range-classifier-v33-p1.onnx"
        _export(model, tensors, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("V33 parity requires CPUExecutionProvider")
        parity_errors: dict[str, float] = {}
        with torch.inference_mode():
            for count in (1, 7, 64, 257):
                sample = tensors[:1].repeat(count, 1, 1, 1)
                expected = model(sample).numpy()
                actual = np.asarray(session.run(["region_logits"], {"region_proposals": sample.numpy()})[0], dtype=np.float32)
                parity_errors[str(count)] = float(np.max(np.abs(expected - actual)))
        phase = "dev_evaluation"

        def run_model(input_values: np.ndarray) -> np.ndarray:
            return np.asarray(session.run(["region_logits"], {"region_proposals": np.ascontiguousarray(input_values, dtype=np.float32)})[0], dtype=np.float32)

        dev_metrics = evaluate_scenes(build_split("dev"), run_model)
        parity_passed = max(parity_errors.values()) <= ONNX_PARITY_TOLERANCE
        passed = bool(
            dev_metrics["precision"] >= float(config["selection_gates"]["precision_minimum"])
            and dev_metrics["recall"] >= float(config["selection_gates"]["recall_minimum"])
            and parity_passed
        )
        report = {
            "schema": "graphreader.ocr-real-range-classifier-v33-result.v1",
            "task": TASK, "revision": REVISION, "candidate_id": "P1",
            "status": "dev_pass" if passed else "dev_failed",
            "architecture": "multiscale-detail-context-geometry-fusion-v1",
            "training_from_scratch": True,
            "model_license": MODEL_LICENSE,
            "source_model_revision": SOURCE_MODEL_REVISION,
            "official_diagnostic_path": OFFICIAL_DIAGNOSTIC_PATH,
            "official_diagnostic_sha256": OFFICIAL_DIAGNOSTIC_SHA256,
            "fine_tune_result_path": FINE_TUNE_RESULT_PATH,
            "fine_tune_result_sha256": FINE_TUNE_RESULT_SHA256,
            "train_split_fingerprint": split_fingerprint("train"),
            "dev_split_fingerprint": split_fingerprint("dev"),
            "synthetic_only": True, "private_or_article_images": False,
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_evidence": evidence, "optimizer_steps": steps,
            "mean_training_loss": sum(loss_trace) / max(1, len(loss_trace)),
            "dev_metrics": dev_metrics,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_errors_by_batch_size": parity_errors,
            "onnx_parity_maximum_absolute_error": max(parity_errors.values()),
            "onnx_parity_tolerance": ONNX_PARITY_TOLERANCE,
            "onnx_parity_passed": parity_passed,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "private_data": False,
            "production_approval": False,
            "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-real-range-classifier-v33-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": "P1",
            "status": "failed_runner", "phase": phase, "optimizer_steps": steps,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "sealed_runs": 0, "private_data": False,
            "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "dev_metrics": report.get("dev_metrics"), "onnx_parity_maximum_absolute_error": report.get("onnx_parity_maximum_absolute_error")}, indent=2, sort_keys=True))
    return 0 if report["status"] == "dev_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
