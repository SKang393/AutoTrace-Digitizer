# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Canonical-budget fine-tune runner for synthetic V32 train/dev data."""

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
from ml.ocr.component_recall_detector_v9.model import ComponentRecallNet
from ml.ocr.component_recall_detector_v9.model_p3 import ScaledComponentRecallNet

from .dataset import build_split, split_fingerprint
from .pipeline import evaluate_scenes
from .protocol import (
    BATCH_SIZE,
    CANONICAL_OUTPUT,
    EPOCHS,
    LEARNING_RATE,
    REVISION,
    SEED,
    SOURCE_CHECKPOINT_PATH,
    SOURCE_CHECKPOINT_SHA256,
    SOURCE_ONNX_PATH,
    SOURCE_ONNX_SHA256,
    SOURCE_MODEL_SEED,
    TASK,
    WEIGHT_DECAY,
)
from .training_data import training_examples


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/real_range_classifier_finetune_v32")
CONFIG_PATH = ROOT / "training/p1.json"
CANONICAL_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    ROOT / "training_data.py",
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/ocr/component_recall_detector_v9/model.py"),
    Path("ml/ocr/component_recall_detector_v9/model_p3.py"),
    Path("ml/ocr/component_recall_detector_v9/protocol.py"),
    Path("ml/ocr/component_spaced_recall_detector_v10/protocol.py"),
    Path("ml/synthetic/dataset.py"),
    Path("ml/synthetic/renderer.py"),
    Path("ml/synthetic/templates.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
    Path("ml/policy/evidence_policy.py"),
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
        raise RuntimeError(f"V32 output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    if config["expected_runner_source_bundle_sha256"] == "ROOT_TO_FILL_SOURCE_BUNDLE_SHA256":
        raise RuntimeError("V32 expected_runner_source_bundle_sha256 must be filled after source review")
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id="P1",
        config_path=CONFIG_PATH, runner_source_paths=CANONICAL_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    optimizer_steps = 0
    phase = "initialization"
    try:
        source_checkpoint = REPO_ROOT / SOURCE_CHECKPOINT_PATH
        if sha256_file(source_checkpoint) != SOURCE_CHECKPOINT_SHA256:
            raise RuntimeError("V10 P2 source checkpoint checksum changed")
        if sha256_file(REPO_ROOT / SOURCE_ONNX_PATH) != SOURCE_ONNX_SHA256:
            raise RuntimeError("V10 P2 source ONNX checksum changed")
        if int(config["source_model_seed"]) != SOURCE_MODEL_SEED:
            raise RuntimeError("V32 source model seed changed")
        values, labels, train_evidence = training_examples()
        train_tensor = torch.from_numpy(values)
        label_tensor = torch.from_numpy(labels)
        model_base = ComponentRecallNet(seed=SOURCE_MODEL_SEED)
        source = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
        model_base.load_state_dict(source["state_dict"])
        model_base.train()
        optimizer = torch.optim.AdamW(model_base.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(config["positive_class_weight"])]))
        order_generator = torch.Generator().manual_seed(SEED)
        loss_trace: list[float] = []
        phase = "training"
        for _ in range(EPOCHS):
            order = torch.randperm(len(label_tensor), generator=order_generator)
            for start in range(0, len(order), BATCH_SIZE):
                indices = order[start : start + BATCH_SIZE]
                logits = model_base(train_tensor.index_select(0, indices))
                loss = criterion(logits, label_tensor.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                loss_trace.append(float(loss.detach()))
        phase = "export"
        model = ScaledComponentRecallNet(model_base.eval()).eval()
        checkpoint_path = output_dir / "graph-text-real-range-classifier-finetune-v32-p1.pt"
        torch.save({"state_dict": model_base.state_dict(), "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256}, checkpoint_path)
        onnx_path = output_dir / "graph-text-real-range-classifier-finetune-v32-p1.onnx"
        _export(model, train_tensor, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("V32 parity requires CPUExecutionProvider")
        parity_rows = []
        for proposal_count in (1, min(256, len(train_tensor))):
            parity_values = train_tensor[:proposal_count]
            with torch.inference_mode():
                expected = model(parity_values).numpy()
            actual = np.asarray(
                session.run(["region_logits"], {"region_proposals": parity_values.numpy()})[0],
                dtype=np.float32,
            )
            parity_rows.append({
                "proposal_count": proposal_count,
                "maximum_absolute_error": float(np.max(np.abs(expected - actual))),
            })
        parity_error = max(float(row["maximum_absolute_error"]) for row in parity_rows)
        dev_scenes = build_split("dev")

        def run_model(input_values: np.ndarray) -> np.ndarray:
            return np.asarray(session.run(["region_logits"], {"region_proposals": np.ascontiguousarray(input_values, dtype=np.float32)})[0], dtype=np.float32)

        phase = "dev_evaluation"
        dev_metrics = evaluate_scenes(dev_scenes, run_model)
        passed = (
            dev_metrics["precision"] >= float(config["selection_gates"]["precision_minimum"])
            and dev_metrics["recall"] >= float(config["selection_gates"]["recall_minimum"])
            and parity_error <= float(config["onnx_parity_tolerance"])
        )
        report = {
            "schema": "graphreader.ocr-real-range-classifier-finetune-v32-result.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": "P1",
            "status": "dev_pass" if passed else "dev_failed",
            "synthetic_only": True,
            "private_or_article_images": False,
            "sealed_public_archive_opened": False,
            "public_gate_evaluations": 0,
            "source_checkpoint_path": SOURCE_CHECKPOINT_PATH.as_posix(),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_onnx_path": SOURCE_ONNX_PATH.as_posix(),
            "source_onnx_sha256": SOURCE_ONNX_SHA256,
            "model_license": config["model_license"],
            "train_split_fingerprint": split_fingerprint("train"),
            "dev_split_fingerprint": split_fingerprint("dev"),
            "training_evidence": train_evidence,
            "optimizer_steps": optimizer_steps,
            "mean_training_loss": sum(loss_trace) / max(1, len(loss_trace)),
            "positive_class_weight": config["positive_class_weight"],
            "dev_metrics": dev_metrics,
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_rows": parity_rows,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_error <= float(config["onnx_parity_tolerance"]),
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
            "schema": "graphreader.ocr-real-range-classifier-finetune-v32-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": "P1",
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "sealed_public_archive_opened": False,
            "public_gate_evaluations": 0,
            "sealed_runs": 0,
            "private_data": False,
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
