# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authorization-ready V17 train/dev runner with metric-aligned labels."""

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

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, seal_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.scale_classifier_v16.train import _configure, _evaluate, _loss
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .protocol import LABEL_POSITIVE_DISTANCE_PX, POSITIVE_OVERSAMPLING_POLICY, V13_MANIFEST_SHA256, V16_RESULT_PATH, V16_RESULT_SHA256

REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-scale-stratified-v17"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/metric_aligned_v17/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/metric_aligned_v17/protocol.py"),
    Path("ml/markers/center/metric_aligned_v17/train.py"),
    Path("ml/markers/center/scale_classifier_v16/model.py"),
    Path("ml/markers/center/scale_classifier_v16/train.py"),
    Path("ml/markers/center/scale_classifier_v16/P1_RESULT.json"),
    Path("ml/markers/center/proposal_geometry_v13/dataset.py"),
    Path("ml/markers/center/proposal_geometry_v13/geometry.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/center/metrics.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _examples(scenes, maximum_negative_per_positive: int, generator: torch.Generator):
    patches, labels, offsets, radii, hard = [], [], [], [], []
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        coordinates = proposals.coordinates
        centers = torch.tensor(scene.centers, dtype=torch.float32)
        scene_radii = torch.tensor(scene.radii, dtype=torch.float32)
        distances = torch.cdist(coordinates, centers)
        nearest, nearest_index = distances.min(dim=1)
        scene_labels = nearest.le(LABEL_POSITIVE_DISTANCE_PX).to(torch.float32)
        scene_hard = torch.zeros(len(coordinates), dtype=torch.bool)
        for point in scene.prohibited:
            scene_hard |= torch.cdist(coordinates, torch.tensor(((point.x, point.y),), dtype=torch.float32)).squeeze(1).le(8.0)
        positive = torch.nonzero(scene_labels > 0.5, as_tuple=False).flatten()
        repeat_counts = 1 + scene_radii.index_select(0, nearest_index).index_select(0, positive).ge(8.0).to(torch.long)
        positive = torch.repeat_interleave(positive, repeat_counts)
        negative = torch.nonzero(scene_labels <= 0.5, as_tuple=False).flatten()
        hard_indices = torch.nonzero(scene_hard & (scene_labels <= 0.5), as_tuple=False).flatten()
        remaining = torch.tensor(sorted(set(negative.tolist()) - set(hard_indices.tolist())), dtype=torch.long)
        budget = max(len(positive) * maximum_negative_per_positive, len(hard_indices))
        random_budget = max(0, budget - len(hard_indices))
        if len(remaining) > random_budget:
            remaining = remaining.index_select(0, torch.randperm(len(remaining), generator=generator)[:random_budget])
        selected = torch.cat((positive, hard_indices, remaining))
        patches.append(proposals.patches.index_select(0, selected))
        labels.append(scene_labels.index_select(0, selected))
        offsets.append((centers.index_select(0, nearest_index) - coordinates).index_select(0, selected) / 4.0)
        radii.append(scene_radii.index_select(0, nearest_index).index_select(0, selected))
        hard.append(scene_hard.index_select(0, selected))
    return torch.cat(patches), torch.cat(labels), torch.cat(offsets), torch.cat(radii), torch.cat(hard)


def _verify_inputs(config: dict[str, object]) -> None:
    previous = REPO_ROOT / V16_RESULT_PATH
    if sha256_file(previous) != V16_RESULT_SHA256:
        raise RuntimeError("V16 P1 result changed")
    payload = json.loads(previous.read_text(encoding="utf-8"))
    if payload.get("status") != "failed_dev_retired_unconsumed" or payload.get("dev_gate_passed") is not False or payload.get("private_data") is not False:
        raise RuntimeError("V17 requires the failed, unconsumed V16 P1 aggregate result")
    if config.get("label_positive_distance_px") != LABEL_POSITIVE_DISTANCE_PX:
        raise RuntimeError("V17 label radius changed")
    if config.get("input_proposal_manifest_sha256") != V13_MANIFEST_SHA256:
        raise RuntimeError("V13 proposal manifest binding changed")


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config_path = REPO_ROOT / CONFIG_PATH
    config = json.loads(config_path.read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID, config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    try:
        _verify_inputs(config)
        _configure(int(config["seed"]))
        manifest_path, manifest_sha256 = seal_manifest(output_dir)
        if manifest_sha256 != V13_MANIFEST_SHA256:
            raise RuntimeError("V13 proposal manifest differs from preregistration")
        train_scenes, dev_scenes = build_selection_scenes("train"), build_selection_scenes("dev")
        generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
        patches, labels, offsets, radii, hard = _examples(train_scenes, int(config["maximum_negative_per_positive"]), generator)
        model = ScaleClassifierNet(ModelConfig(seed=int(config["seed"])))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        model.train()
        optimizer_steps = 0
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(labels), generator=generator)
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                loss = _loss(model.forward_raw(patches.index_select(0, indices)), labels.index_select(0, indices), offsets.index_select(0, indices), radii.index_select(0, indices), hard.index_select(0, indices), positive_weight=float(config["positive_loss_weight"]), hard_weight=float(config["hard_negative_loss_weight"]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
        model.eval()
        comparisons = [_evaluate(dev_scenes, model, float(threshold)) for threshold in config["selection_thresholds"]]
        selected = max(comparisons, key=lambda row: (min(float(row["precision"]), float(row["recall"])), float(row["f1"])))
        dev_passed = float(selected["precision"]) >= float(config["acceptance_bar"]["precision_minimum"]) and float(selected["recall"]) >= float(config["acceptance_bar"]["recall_minimum"])
        checkpoint = output_dir / "marker-center-metric-aligned-v17-p1.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.export_contract(), "selected_threshold": selected["threshold"], "dataset_manifest_sha256": manifest_sha256, "label_positive_distance_px": LABEL_POSITIVE_DISTANCE_PX}, checkpoint)
        onnx_path = output_dir / "marker-center-metric-aligned-v17-p1.onnx"
        example = torch.zeros((1, 3, 33, 33), dtype=torch.float32)
        torch.onnx.export(model, example, onnx_path, input_names=["candidate_patches"], output_names=["candidate_predictions"], dynamic_axes={"candidate_patches": {0: "candidate_count"}, "candidate_predictions": {0: "candidate_count"}}, opset_version=18, dynamo=False)
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("V17 parity requires CPUExecutionProvider")
        parity_rows = []
        for count in config["onnx_dynamic_candidate_counts"]:
            value = torch.zeros((int(count), 3, 33, 33), dtype=torch.float32)
            with torch.inference_mode():
                expected = model(value).numpy()
            actual = session.run(["candidate_predictions"], {"candidate_patches": value.numpy()})[0]
            parity_rows.append({"candidate_count": int(count), "maximum_absolute_error": float(np.max(np.abs(expected - actual)))})
        parity = max(float(row["maximum_absolute_error"]) for row in parity_rows)
        report = {"schema": "graphreader.marker-center-scale-stratified-candidate.v17", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "dev_passed" if dev_passed and parity <= float(config["onnx_parity_tolerance"]) else "failed_dev", "synthetic_only": True, "private_data": False, "private_or_article_images": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "sealed_runs": 0, "training_authorization": authorization.binding, "v16_result_path": V16_RESULT_PATH, "v16_result_sha256": V16_RESULT_SHA256, "model_license": config["model_license"], "label_positive_distance_px": LABEL_POSITIVE_DISTANCE_PX, "positive_oversampling_policy": POSITIVE_OVERSAMPLING_POLICY, "dataset_manifest_sha256": manifest_sha256, "training_example_count": len(labels), "hard_negative_example_count": int(hard.sum()), "optimizer_steps": optimizer_steps, "dev_comparisons": comparisons, "selected": selected, "dev_gate_passed": dev_passed, "checkpoint_sha256": sha256_file(checkpoint), "onnx_sha256": sha256_file(onnx_path), "onnx_provider": "CPUExecutionProvider", "onnx_dynamic_candidate_counts": parity_rows, "onnx_parity_maximum_absolute_error": parity, "onnx_parity_tolerance": config["onnx_parity_tolerance"], "onnx_parity_passed": parity <= float(config["onnx_parity_tolerance"]), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "production_approval": False, "release_eligible": False}
    except Exception as error:
        report = {"schema": "graphreader.marker-center-metric-aligned-failure.v17", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner", "exception_type": type(error).__name__, "exception_message": str(error), "synthetic_only": True, "private_data": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "sealed_runs": 0, "training_authorization": authorization.binding, "completed_utc": datetime.now(timezone.utc).isoformat()}
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
