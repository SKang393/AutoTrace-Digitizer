# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 training with normalized shape and explicit geometry."""

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

from .dataset import build_split, split_fingerprint
from .p3_model import ScaleAwareComponentGeometricGlyphNet
from .p3_pipeline import evaluate_samples, glyph_training_examples
from .protocol import (
    MARKER_EXCLUSION_ACCURACY_MINIMUM,
    ROLE_ACCURACY_MINIMUM,
    TASK,
    VALIDATION_EXACT_MATCH_MINIMUM,
    REVISION,
)
from .train_p1 import _configure, _export, _runner


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/ocr/component_geometric_v4/training/p3.json")
CANONICAL_OUTPUT = Path("ml/ocr/component_geometric_v4/artifacts/P3-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/component_geometric_v4/dataset.py"),
    Path("ml/ocr/component_geometric_v4/model.py"),
    Path("ml/ocr/component_geometric_v4/p3_dataset.py"),
    Path("ml/ocr/component_geometric_v4/p3_model.py"),
    Path("ml/ocr/component_geometric_v4/p3_pipeline.py"),
    Path("ml/ocr/component_geometric_v4/protocol.py"),
    Path("ml/ocr/component_geometric_v4/train_p1.py"),
    Path("ml/ocr/component_geometric_v4/train_p3.py"),
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
        seed = int(config["seed"])
        generator = _configure(seed)
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection manifest checksum does not match P3 preregistration")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        sealed_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(sealed_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Sealed-public seal checksum does not match P3 preregistration")
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / sealed["fixture_archive_path"]) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("Sealed-public archive changed before P3 training")
        training_samples = build_split("train")
        validation_samples = build_split("validation")
        if split_fingerprint(training_samples) != selection["train_split_fingerprint"]:
            raise RuntimeError("P3 training renderer no longer reproduces the frozen split")
        if split_fingerprint(validation_samples) != selection["validation_split_fingerprint"]:
            raise RuntimeError("P3 validation renderer no longer reproduces the frozen split")
        training_values, training_labels = glyph_training_examples(training_samples)
        validation_values, _ = glyph_training_examples(validation_samples)
        model = ScaleAwareComponentGeometricGlyphNet(seed=seed)
        phase = "onnx_preflight"
        preflight_path = output_dir / "export-preflight.onnx"
        _export(model.eval(), torch.from_numpy(validation_values[:8]), preflight_path)
        preflight_sha256 = sha256_file(preflight_path)
        preflight_path.unlink()
        phase = "training"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()
        values = torch.from_numpy(training_values)
        labels = torch.from_numpy(training_labels)
        batch_size = int(config["batch_size"])
        loss_checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(labels), generator=generator)
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
                loss_checkpoints.append({"epoch": epoch + 1, "cross_entropy": sum(losses) / len(losses)})
        phase = "selection"
        model.eval()
        threshold_results = [
            {"threshold": float(threshold), "metrics": evaluate_samples(validation_samples, _runner(model), float(threshold))}
            for threshold in config["selection_thresholds"]
        ]
        selected = max(
            threshold_results,
            key=lambda item: (
                item["metrics"]["exact_match"],
                item["metrics"]["marker_exclusion_accuracy"],
                item["metrics"]["role_accuracy"],
                -item["metrics"]["character_error_rate"],
                item["threshold"],
            ),
        )
        metrics = selected["metrics"]
        selection_passed = (
            metrics["exact_match"] >= VALIDATION_EXACT_MATCH_MINIMUM
            and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
            and metrics["marker_exclusion_accuracy"] >= MARKER_EXCLUSION_ACCURACY_MINIMUM
        )
        phase = "export"
        checkpoint_path = output_dir / "graph-numeric-component-geometric-v4-p3.pt"
        torch.save({"state_dict": model.state_dict(), "selected_threshold": selected["threshold"]}, checkpoint_path)
        onnx_path = output_dir / "graph-numeric-component-geometric-v4-p3.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"glyphs": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        report: dict[str, object] = {
            "schema": "graphreader.ocr-component-geometric-training-report.v1",
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
            "training_authorization": authorization.binding,
            "training_sample_count": len(training_samples),
            "training_glyph_count": len(training_labels),
            "validation_sample_count": len(validation_samples),
            "epochs": config["epochs"],
            "seed": seed,
            "optimizer_steps": optimizer_steps,
            "loss_checkpoints": loss_checkpoints,
            "onnx_preflight_sha256": preflight_sha256,
            "threshold_comparisons": threshold_results,
            "selected_threshold": selected["threshold"],
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
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-component-geometric-training-failure.v1",
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
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
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
    summary = {
        "candidate_id": report["candidate_id"],
        "status": report["status"],
        "selected_threshold": report["selected_threshold"],
        "selection_gate_passed": report["selection_gate_passed"],
        "onnx_parity_passed": report["onnx_parity_passed"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
        "report_path": (REPO_ROOT / arguments.output / "candidate-report.json").relative_to(REPO_ROOT).as_posix(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
