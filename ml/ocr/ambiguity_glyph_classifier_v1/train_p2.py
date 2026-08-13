# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use profile-aware P2 training on the unchanged frozen split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_partition, split_fingerprint
from .model_p2 import ProfileAwareAmbiguityGlyphNet
from .protocol import GATES, GLYPHS, REVISION, TASK
from .train_p1 import _configure, _export, _metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/ambiguity_glyph_classifier_v1")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model_p2.py", ROOT / "protocol.py", ROOT / "train_p1.py", ROOT / "train_p2.py",
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Ambiguity glyph P2 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
                                                config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    optimizer_steps = 0
    try:
        seed = int(config["seed"])
        generator = _configure(seed)
        for key, label in (("selection_manifest", "selection manifest"), ("sealed_public_test_seal", "public seal"),
                           ("p1_result", "P1 failure")):
            path = REPO_ROOT / str(config[f"{key}_path"])
            if sha256_file(path) != config[f"{key}_sha256"]:
                raise RuntimeError(f"Ambiguity glyph P2 {label} changed")
        selection = json.loads((REPO_ROOT / str(config["selection_manifest_path"])).read_text(encoding="utf-8"))
        public_seal = json.loads((REPO_ROOT / str(config["sealed_public_test_seal_path"])).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / public_seal["fixture_archive_path"]) != public_seal["fixture_archive_sha256"]:
            raise RuntimeError("Ambiguity glyph P2 public archive changed before training")
        _, _, train_values, train_labels = build_partition("train")
        _, _, validation_values, validation_labels = build_partition("validation")
        if split_fingerprint("train") != selection["train_split_fingerprint"] or split_fingerprint("validation") != selection["validation_split_fingerprint"]:
            raise RuntimeError("Ambiguity glyph P2 frozen split changed")
        model = ProfileAwareAmbiguityGlyphNet(seed=seed)
        phase = "onnx_preflight"
        preflight = output_dir / "export-preflight.onnx"
        _export(model.eval(), torch.from_numpy(validation_values[:8]), preflight)
        preflight_sha256 = sha256_file(preflight)
        preflight.unlink()
        phase = "training"
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                      weight_decay=float(config["weight_decay"]))
        criterion = nn.CrossEntropyLoss()
        values = torch.from_numpy(train_values)
        labels = torch.from_numpy(train_labels)
        batch_size = int(config["batch_size"])
        checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(labels), generator=generator)
            losses: list[float] = []
            for start in range(0, len(labels), batch_size):
                indices = order[start:start + batch_size]
                loss = criterion(model(values.index_select(0, indices)), labels.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({"epoch": epoch + 1, "cross_entropy": sum(losses) / len(losses)})
        phase = "selection"
        model.eval()
        with torch.inference_mode():
            validation_logits = model(torch.from_numpy(validation_values)).numpy()
        metrics = _metrics(validation_logits, validation_labels)
        selection_passed = (
            metrics["accuracy"] >= GATES["validation_accuracy_minimum"]
            and metrics["macro_accuracy"] >= GATES["validation_macro_accuracy_minimum"]
            and min(metrics["per_class_accuracy"].values()) >= GATES["validation_per_class_accuracy_minimum"]
        )
        phase = "export"
        checkpoint_path = output_dir / "graph-ambiguity-glyph-classifier-v1-p2.pt"
        torch.save({"state_dict": model.state_dict(), "class_order": list(GLYPHS)}, checkpoint_path)
        onnx_path = output_dir / "graph-ambiguity-glyph-classifier-v1-p2.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"glyphs": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= GATES["onnx_parity_maximum_absolute_error"]
        report: dict[str, object] = {
            "schema": "graphreader.ocr-ambiguity-glyph-training-report.v1", "task": TASK, "revision": REVISION,
            "candidate_id": CANDIDATE_ID, "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "production_approval": False, "release_eligible": False, "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False, "synthetic_only": True, "private_or_article_images": False,
            "chandler_included": False, "training_authorization": authorization.binding, "class_order": list(GLYPHS),
            "p1_result_sha256": config["p1_result_sha256"], "training_sample_count": len(train_labels),
            "validation_sample_count": len(validation_labels), "epochs": config["epochs"], "seed": seed,
            "optimizer_steps": optimizer_steps, "loss_checkpoints": checkpoints, "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(), "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size, "onnx_provider": "CPUExecutionProvider",
            "onnx_preflight_sha256": preflight_sha256, "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_passed": parity_passed, "gate_requirements": GATES,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {"schema": "graphreader.ocr-ambiguity-glyph-training-failure.v1", "task": TASK, "revision": REVISION,
                   "candidate_id": CANDIDATE_ID, "status": "failed_runner", "phase": phase,
                   "optimizer_steps": optimizer_steps, "exception_type": type(error).__name__,
                   "exception_message": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                   "training_authorization": authorization.binding, "public_gate_evaluations": 0,
                   "sealed_public_archive_opened": False, "production_approval": False, "release_eligible": False}
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "selection_metrics": report["selection_metrics"],
                      "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
