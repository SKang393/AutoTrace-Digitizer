# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for the preregistered OCR component-fusion detector V8."""

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

from .dataset import build_split, proposal_examples, split_fingerprint
from .model import ComponentFusionNet
from .pipeline import evaluate_scenes
from .protocol import REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/ocr/component_fusion_detector_v8/training/p1.json")
CANONICAL_OUTPUT = Path("ml/ocr/component_fusion_detector_v8/artifacts/P1-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/component_fusion_detector_v8/dataset.py"),
    Path("ml/ocr/component_fusion_detector_v8/model.py"),
    Path("ml/ocr/component_fusion_detector_v8/pipeline.py"),
    Path("ml/ocr/component_fusion_detector_v8/protocol.py"),
    Path("ml/ocr/component_fusion_detector_v8/train_p1.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.Generator().manual_seed(seed)


def _runner(model: nn.Module):
    def run(values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(values)).numpy()

    return run


def _balanced_order(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    negative = torch.nonzero(labels == 0, as_tuple=False).flatten()
    positive = torch.nonzero(labels == 1, as_tuple=False).flatten()
    if len(negative) == 0 or len(positive) == 0:
        raise RuntimeError("OCR V8 balanced sampler requires both classes")
    negative = negative.index_select(0, torch.randperm(len(negative), generator=generator))
    positive = positive.index_select(0, torch.randperm(len(positive), generator=generator))
    target = max(len(negative), len(positive))

    def expand(values: torch.Tensor) -> torch.Tensor:
        repeats = (target + len(values) - 1) // len(values)
        return values.repeat(repeats)[:target]

    paired = torch.stack((expand(negative), expand(positive)), dim=1).flatten()
    return paired.index_select(0, torch.randperm(len(paired), generator=generator))


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["region_proposals"],
        output_names=["region_logits"],
        dynamic_axes={"region_proposals": {0: "proposal_count"}, "region_logits": {0: "proposal_count"}},
        opset_version=18,
        dynamo=False,
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
            raise RuntimeError("OCR V8 selection manifest checksum changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        sealed_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(sealed_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("OCR V8 public seal checksum changed")
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / sealed["fixture_archive_path"]) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("OCR V8 sealed-public archive changed before training")
        training_scenes = build_split("train")
        validation_scenes = build_split("validation")
        if split_fingerprint(training_scenes) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V8 training renderer changed after freeze")
        if split_fingerprint(validation_scenes) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V8 validation renderer changed after freeze")
        training_values, training_labels = proposal_examples(training_scenes)
        validation_values, _ = proposal_examples(validation_scenes)
        model = ComponentFusionNet(seed=seed)
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
        checkpoints: list[dict[str, float | int]] = []
        model.train()
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
        comparisons = [
            {"threshold": float(threshold), "metrics": evaluate_scenes(validation_scenes, _runner(model), float(threshold))}
            for threshold in config["selection_thresholds"]
        ]
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
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_region_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )
        phase = "export"
        checkpoint_path = output_dir / "graph-text-component-fusion-v8-p1.pt"
        torch.save({"state_dict": model.state_dict(), "selected_threshold": selected["threshold"]}, checkpoint_path)
        onnx_path = output_dir / "graph-text-component-fusion-v8-p1.onnx"
        parity_values = torch.from_numpy(validation_values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        report: dict[str, object] = {
            "schema": "graphreader.ocr-component-fusion-training-report.v1",
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
            "predecessor_public_cases_used_for_selection": False,
            "prior_public_sample_or_pixel_inspection_used": False,
            "training_authorization": authorization.binding,
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
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-component-fusion-training-failure.v1",
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
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "status": report["status"],
                "selected_threshold": report["selected_threshold"],
                "selection_gate_passed": report["selection_gate_passed"],
                "onnx_parity_passed": report["onnx_parity_passed"],
                "report_path": (REPO_ROOT / arguments.output / "candidate-report.json").relative_to(REPO_ROOT).as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
