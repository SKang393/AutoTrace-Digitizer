# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for the source-group ambiguity classifier."""

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
from .model import SourceGroupAmbiguityNet
from .protocol import CANDIDATE_ID, GATES, GLYPHS, REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/ambiguity_source_group_classifier_v3")
CONFIG_PATH = ROOT / "training/p1.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "crop.py", ROOT / "dataset.py", ROOT / "model.py", ROOT / "protocol.py", ROOT / "train_p1.py",
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.Generator().manual_seed(seed)


def _metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    predictions = np.argmax(logits, axis=1)
    per_class = {
        glyph: float(np.mean(predictions[labels == index] == index))
        for index, glyph in enumerate(GLYPHS)
    }
    return {
        "sample_count": len(labels),
        "correct_count": int(np.sum(predictions == labels)),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_accuracy": float(np.mean(list(per_class.values()))),
        "per_class_accuracy": per_class,
    }


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model, example, path, input_names=["glyphs"], output_names=["logits"],
        dynamic_axes={"glyphs": {0: "glyph_count"}, "logits": {0: "glyph_count"}},
        opset_version=18, dynamo=False,
    )


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Source-group ambiguity output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    optimizer_steps = 0
    try:
        seed = int(config["seed"])
        generator = _configure(seed)
        for key, label in (
            ("selection_manifest", "selection manifest"),
            ("sealed_public_test_seal", "sealed public test seal"),
            ("trigger_result", "trigger result"),
        ):
            path = REPO_ROOT / str(config[f"{key}_path"])
            if sha256_file(path) != config[f"{key}_sha256"]:
                raise RuntimeError(f"Source-group ambiguity {label} changed")
        selection = json.loads((REPO_ROOT / str(config["selection_manifest_path"])).read_text(encoding="utf-8"))
        public_seal = json.loads((REPO_ROOT / str(config["sealed_public_test_seal_path"])).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / str(public_seal["fixture_archive_path"])) != public_seal["fixture_archive_sha256"]:
            raise RuntimeError("Source-group ambiguity public archive changed before training")
        _, _, train_values, train_labels = build_partition("train")
        _, _, validation_values, validation_labels = build_partition("validation")
        if split_fingerprint("train") != selection["train_split_fingerprint"]:
            raise RuntimeError("Source-group ambiguity training split changed")
        if split_fingerprint("validation") != selection["validation_split_fingerprint"]:
            raise RuntimeError("Source-group ambiguity validation split changed")

        model = SourceGroupAmbiguityNet(seed=seed)
        phase = "onnx_preflight"
        preflight = output_dir / "export-preflight.onnx"
        _export(model.eval(), torch.from_numpy(validation_values[:8]), preflight)
        preflight_sha256 = sha256_file(preflight)
        preflight.unlink()
        phase = "training"
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
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
        checkpoint_path = output_dir / "graph-ambiguity-source-group-v3-p1.pt"
        torch.save({"state_dict": model.state_dict(), "class_order": list(GLYPHS)}, checkpoint_path)
        onnx_path = output_dir / "graph-ambiguity-source-group-v3-p1.onnx"
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
            "schema": "graphreader.ocr-ambiguity-source-group-training-report.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "production_approval": False, "release_eligible": False,
            "public_gate_evaluations": 0, "sealed_public_archive_opened": False,
            "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
            "training_authorization": authorization.binding, "class_order": list(GLYPHS),
            "training_sample_count": len(train_labels), "validation_sample_count": len(validation_labels),
            "epochs": config["epochs"], "seed": seed, "optimizer_steps": optimizer_steps,
            "loss_checkpoints": checkpoints, "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider", "onnx_preflight_sha256": preflight_sha256,
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
            "schema": "graphreader.ocr-ambiguity-source-group-training-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": optimizer_steps,
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
