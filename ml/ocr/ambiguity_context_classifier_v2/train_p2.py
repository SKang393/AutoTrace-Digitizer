# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-training P2 parity repair over the exact P1 weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_partition, split_fingerprint
from .model import LineContextAmbiguityNet
from .model_p2 import ParityScaledLineContextNet
from .protocol import GATES, GLYPHS, REVISION, TASK
from .train_p1 import _configure, _export, _metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/ambiguity_context_classifier_v2")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "model_p2.py", ROOT / "protocol.py",
    ROOT / "train_p1.py", ROOT / "train_p2.py",
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Line-context ambiguity P2 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    try:
        seed = int(config["seed"])
        _configure(seed)
        for key, label in (
            ("selection_manifest", "selection manifest"),
            ("sealed_public_test_seal", "sealed public test seal"),
            ("p1_result", "P1 result"),
            ("p1_checkpoint", "P1 checkpoint"),
        ):
            path = REPO_ROOT / str(config[f"{key}_path"])
            if sha256_file(path) != config[f"{key}_sha256"]:
                raise RuntimeError(f"Line-context ambiguity P2 {label} changed")
        selection = json.loads((REPO_ROOT / str(config["selection_manifest_path"])).read_text(encoding="utf-8"))
        public_seal = json.loads((REPO_ROOT / str(config["sealed_public_test_seal_path"])).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / str(public_seal["fixture_archive_path"])) != public_seal["fixture_archive_sha256"]:
            raise RuntimeError("Line-context ambiguity public archive changed before P2")
        _, _, validation_values, validation_labels = build_partition("validation")
        if split_fingerprint("validation") != selection["validation_split_fingerprint"]:
            raise RuntimeError("Line-context ambiguity P2 validation split changed")

        phase = "checkpoint_load"
        checkpoint = torch.load(REPO_ROOT / str(config["p1_checkpoint_path"]), map_location="cpu", weights_only=True)
        if checkpoint.get("class_order") != list(GLYPHS):
            raise RuntimeError("Line-context ambiguity P1 checkpoint class order changed")
        base = LineContextAmbiguityNet(seed=seed)
        base.load_state_dict(checkpoint["state_dict"])
        model = ParityScaledLineContextNet(base).eval()
        phase = "selection"
        with torch.inference_mode():
            validation_logits = model(torch.from_numpy(validation_values)).numpy()
        metrics = _metrics(validation_logits, validation_labels)
        selection_passed = (
            metrics["accuracy"] >= GATES["validation_accuracy_minimum"]
            and metrics["macro_accuracy"] >= GATES["validation_macro_accuracy_minimum"]
            and min(metrics["per_class_accuracy"].values()) >= GATES["validation_per_class_accuracy_minimum"]
        )
        phase = "export"
        onnx_path = output_dir / "graph-ambiguity-line-context-v2-p2.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"glyphs": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        mismatch_count = int(np.sum(np.argmax(expected, axis=1) != np.argmax(actual, axis=1)))
        parity_passed = (
            maximum_error <= GATES["onnx_parity_maximum_absolute_error"]
            and mismatch_count == GATES["onnx_argmax_mismatch_count"]
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-ambiguity-context-training-report.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "isolated_change": "multiply exact P1 output logits by 0.5 during export",
            "optimizer_steps": 0, "weights_changed": False,
            "p1_checkpoint_path": config["p1_checkpoint_path"],
            "p1_checkpoint_sha256": config["p1_checkpoint_sha256"],
            "production_approval": False, "release_eligible": False,
            "public_gate_evaluations": 0, "sealed_public_archive_opened": False,
            "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
            "training_authorization": authorization.binding, "class_order": list(GLYPHS),
            "validation_sample_count": len(validation_labels), "seed": seed,
            "selection_metrics": metrics, "selection_gate_passed": selection_passed,
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_argmax_mismatch_count": mismatch_count, "onnx_parity_passed": parity_passed,
            "gate_requirements": GATES,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-ambiguity-context-training-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": 0,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
            "public_gate_evaluations": 0, "sealed_public_archive_opened": False,
            "production_approval": False, "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"], "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
        "onnx_argmax_mismatch_count": report["onnx_argmax_mismatch_count"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
