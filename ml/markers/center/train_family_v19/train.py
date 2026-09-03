# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authorization-ready V19 runner; execution is intentionally deferred."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.markers.center.line_aware_v1.pipeline import postprocess_predictions
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, seal_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.scale_classifier_v16.train import _evaluate, _loss
from ml.markers.center.metric_aligned_v17.train import _examples as v17_examples
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate

from .protocol import (
    V13_MANIFEST_SHA256,
    V18_RESULT_PATH,
    V18_RESULT_SHA256,
)
from .training_families import build_train_scenes

REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-train-family-v19"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/train_family_v19/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/train_family_v19/protocol.py"),
    Path("ml/markers/center/train_family_v19/training_families.py"),
    Path("ml/markers/center/train_family_v19/train.py"),
    Path("ml/markers/center/hard_positive_v18/P1_RESULT.json"),
    Path("ml/markers/center/metric_aligned_v17/train.py"),
    Path("ml/markers/center/scale_classifier_v16/model.py"),
    Path("ml/markers/center/scale_classifier_v16/train.py"),
    Path("ml/markers/center/proposal_geometry_v13/dataset.py"),
    Path("ml/markers/center/proposal_geometry_v13/geometry.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def run(output_dir: Path) -> dict[str, object]:
    """Run one authorized synthetic candidate after the root integrator enables it."""
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
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
        previous = REPO_ROOT / V18_RESULT_PATH
        if sha256_file(previous) != V18_RESULT_SHA256:
            raise RuntimeError("V18 aggregate result changed")
        previous_result = json.loads(previous.read_text(encoding="utf-8"))
        if previous_result.get("dev_gate_passed") is not False:
            raise RuntimeError("V19 requires the failed V18 dev result")
        _configure(int(config["seed"]))
        _, manifest_sha256 = seal_manifest(output_dir)
        if manifest_sha256 != V13_MANIFEST_SHA256:
            raise RuntimeError("V13 proposal manifest changed")
        train_scenes = build_train_scenes()
        if len(train_scenes) != int(config["train_scene_count"]):
            raise RuntimeError("V19 synthetic train-family count changed")
        generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
        patches, labels, offsets, radii, hard = v17_examples(
            train_scenes, int(config["maximum_negative_per_positive"]), generator,
        )
        model = ScaleClassifierNet(ModelConfig(seed=int(config["seed"])))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        optimizer_steps = 0
        phase = "training"
        model.train()
        for _ in range(int(config["epochs"])):
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                loss = _loss(
                    model.forward_raw(patches.index_select(0, indices)),
                    labels.index_select(0, indices), offsets.index_select(0, indices),
                    radii.index_select(0, indices), hard.index_select(0, indices),
                    positive_weight=float(config["positive_loss_weight"]),
                    hard_weight=float(config["hard_negative_loss_weight"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
        phase = "dev"
        dev_scenes = build_selection_scenes("dev")
        model.eval()
        comparisons = [_evaluate(dev_scenes, model, float(value)) for value in config["selection_thresholds"]]
        selected = max(comparisons, key=lambda row: (min(float(row["precision"]), float(row["recall"])), float(row["f1"])))
        dev_passed = float(selected["precision"]) >= float(config["acceptance_bar"]["precision_minimum"]) and float(selected["recall"]) >= float(config["acceptance_bar"]["recall_minimum"])
        phase = "export"
        checkpoint = output_dir / "marker-center-train-family-v19-p1.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.export_contract(), "selected_threshold": selected["threshold"], "dataset_manifest_sha256": manifest_sha256}, checkpoint)
        onnx_path = output_dir / "marker-center-train-family-v19-p1.onnx"
        example = torch.zeros((1, 3, 33, 33), dtype=torch.float32)
        torch.onnx.export(model, example, onnx_path, input_names=["candidate_patches"], output_names=["candidate_predictions"], dynamic_axes={"candidate_patches": {0: "candidate_count"}, "candidate_predictions": {0: "candidate_count"}}, opset_version=18, dynamo=False)
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("V19 parity requires CPUExecutionProvider")
        parity_rows = []
        for count in config["onnx_dynamic_candidate_counts"]:
            value = torch.zeros((int(count), 3, 33, 33), dtype=torch.float32)
            with torch.inference_mode():
                expected = model(value).numpy()
            actual = session.run(["candidate_predictions"], {"candidate_patches": value.numpy()})[0]
            parity_rows.append({"candidate_count": int(count), "maximum_absolute_error": float(np.max(np.abs(expected - actual)))})
        parity = max(float(row["maximum_absolute_error"]) for row in parity_rows)
        report = {"schema": "graphreader.marker-center-train-family-candidate.v19", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "dev_passed" if dev_passed and parity <= float(config["onnx_parity_tolerance"]) else "failed_dev", "synthetic_only": True, "private_data": False, "private_or_article_images": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "sealed_runs": 0, "training_authorization": authorization.binding, "v18_result_path": V18_RESULT_PATH, "v18_result_sha256": V18_RESULT_SHA256, "model_license": config["model_license"], "dataset_manifest_sha256": manifest_sha256, "train_scene_count": len(train_scenes), "training_example_count": len(labels), "hard_negative_example_count": int(hard.sum()), "optimizer_steps": optimizer_steps, "dev_comparisons": comparisons, "selected": selected, "dev_gate_passed": dev_passed, "checkpoint_sha256": sha256_file(checkpoint), "onnx_sha256": sha256_file(onnx_path), "onnx_provider": "CPUExecutionProvider", "onnx_dynamic_candidate_counts": parity_rows, "onnx_parity_maximum_absolute_error": parity, "onnx_parity_tolerance": config["onnx_parity_tolerance"], "onnx_parity_passed": parity <= float(config["onnx_parity_tolerance"]), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "production_approval": False, "release_eligible": False}
    except Exception as error:
        report = {"schema": "graphreader.marker-center-train-family-failure.v19", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner", "phase": phase, "exception_type": type(error).__name__, "exception_message": str(error), "synthetic_only": True, "private_data": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "training_authorization": authorization.binding, "completed_utc": datetime.now(timezone.utc).isoformat()}
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output_dir.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "dev_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
