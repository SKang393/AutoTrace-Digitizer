# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authorization-ready V15 fine-tuning runner; execution is intentionally deferred."""

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
import torch.nn.functional as functional

from ml.markers.center.metrics import aggregate_scene_metrics, center_metrics
from ml.markers.center.line_aware_v1.pipeline import MarkerPrediction, postprocess_predictions
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, seal_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.radial_feature_v1.model import RadialFeatureModelConfig, RadialFeatureNet
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate

REPO_ROOT = Path(__file__).resolve().parents[4]
REVISION = "marker-center-geometry-finetune-v15"
TASK = "marker-center"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/geometry_finetune_v15/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/geometry_finetune_v15/protocol.py"),
    Path("ml/markers/center/geometry_finetune_v15/train.py"),
    Path("ml/markers/center/proposal_geometry_v13/dataset.py"),
    Path("ml/markers/center/proposal_geometry_v13/geometry.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/center/radial_feature_v1/model.py"),
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


def _examples(scenes, maximum_negative_per_positive: int, generator: torch.Generator):
    patches, labels, offsets, radii, hard = [], [], [], [], []
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        coordinates = proposals.coordinates
        centers = torch.tensor(scene.centers, dtype=torch.float32)
        scene_radii = torch.tensor(scene.radii, dtype=torch.float32)
        distances = torch.cdist(coordinates, centers)
        nearest, nearest_index = distances.min(dim=1)
        scene_labels = nearest.le(3.0).to(torch.float32)
        scene_hard = torch.zeros(len(coordinates), dtype=torch.bool)
        for point in scene.prohibited:
            scene_hard |= torch.cdist(coordinates, torch.tensor(((point.x, point.y),), dtype=torch.float32)).squeeze(1).le(8.0)
        positive = torch.nonzero(scene_labels > 0.5, as_tuple=False).flatten()
        negative = torch.nonzero(scene_labels <= 0.5, as_tuple=False).flatten()
        hard_indices = torch.nonzero(scene_hard & (scene_labels <= 0.5), as_tuple=False).flatten()
        remaining = torch.tensor(sorted(set(negative.tolist()) - set(hard_indices.tolist())), dtype=torch.long)
        budget = max(len(positive) * maximum_negative_per_positive, len(hard_indices))
        if len(remaining) > max(0, budget - len(hard_indices)):
            remaining = remaining.index_select(0, torch.randperm(len(remaining), generator=generator)[: max(0, budget - len(hard_indices))])
        selected = torch.cat((positive, hard_indices, remaining)).unique(sorted=True)
        patches.append(proposals.patches.index_select(0, selected))
        labels.append(scene_labels.index_select(0, selected))
        offsets.append((centers.index_select(0, nearest_index) - coordinates).index_select(0, selected) / 4.0)
        radii.append(scene_radii.index_select(0, nearest_index).index_select(0, selected))
        hard.append(scene_hard.index_select(0, selected))
    return torch.cat(patches), torch.cat(labels), torch.cat(offsets), torch.cat(radii), torch.cat(hard)


def _loss(raw, labels, offsets, radii, hard, *, positive_weight: float, hard_weight: float) -> torch.Tensor:
    weights = torch.where(hard, torch.full_like(labels, hard_weight), torch.ones_like(labels))
    weights = torch.where(labels > 0.5, torch.full_like(labels, positive_weight), weights)
    classification = (functional.binary_cross_entropy_with_logits(raw[:, 0], labels, reduction="none") * weights).mean()
    positive = labels > 0.5
    if torch.any(positive):
        offset = functional.smooth_l1_loss(torch.tanh(raw[positive, 1:3]) * 0.75, offsets[positive])
        radius = functional.smooth_l1_loss(2.5 + torch.sigmoid(raw[positive, 3]) * 5.5, radii[positive].clamp(2.5, 8.0))
    else:
        offset, radius = raw[:, 1:3].sum() * 0, raw[:, 3].sum() * 0
    return classification + 1.25 * offset + 0.25 * radius


def _runner(model):
    def run(value: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(np.asarray(value, dtype=np.float32))).numpy()
    return run


def _evaluate(scenes, model, threshold: float) -> dict[str, object]:
    values = []
    prohibited = 0
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        with torch.inference_mode():
            output = model(proposals.patches).numpy()
        predictions = postprocess_predictions(scene, proposals, output, threshold=threshold)
        values.append(center_metrics(predictions, scene.centers, 5.0))
        prohibited += sum(any(((prediction.x - point.x) ** 2 + (prediction.y - point.y) ** 2) <= 25.0 for prediction in predictions) for point in scene.prohibited)
    aggregate = aggregate_scene_metrics(values, 5.0)
    return {"threshold": threshold, "scene_count": len(scenes), "true_positives": aggregate.true_positives, "false_positives": aggregate.false_positives, "false_negatives": aggregate.false_negatives, "duplicate_count": aggregate.duplicate_count, "precision": aggregate.precision, "recall": aggregate.recall, "f1": aggregate.f1, "prohibited_structure_hits": prohibited}


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID, config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    phase = "initialization"
    try:
        if config["expected_runner_source_bundle_sha256"] == "REPLACE_BY_ROOT_INTEGRATOR":
            raise RuntimeError("Root integrator must replace expected_runner_source_bundle_sha256 after committing V15 sources")
        _configure(int(config["seed"]))
        source = REPO_ROOT / config["source_checkpoint_path"]
        if sha256_file(source) != config["source_checkpoint_sha256"]:
            raise RuntimeError("Checksum-bound runtime-consistency P2 checkpoint changed")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        model = RadialFeatureNet(RadialFeatureModelConfig(**payload["config"]["model"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        phase = "dataset"
        manifest_path, manifest_sha256 = seal_manifest(output_dir)
        if manifest_sha256 != config["input_proposal_manifest_sha256"]:
            raise RuntimeError("V13 proposal manifest does not match the preregistered checksum")
        train_scenes, dev_scenes = build_selection_scenes("train"), build_selection_scenes("dev")
        generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
        patches, labels, offsets, radii, hard = _examples(train_scenes, int(config["maximum_negative_per_positive"]), generator)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        phase = "training"
        optimizer_steps = 0
        model.train()
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
        phase = "dev"
        model.eval()
        comparisons = [_evaluate(dev_scenes, model, float(threshold)) for threshold in config["selection_thresholds"]]
        selected = max(comparisons, key=lambda row: (min(float(row["precision"]), float(row["recall"])), float(row["f1"])))
        dev_passed = float(selected["precision"]) >= float(config["acceptance_bar"]["precision_minimum"]) and float(selected["recall"]) >= float(config["acceptance_bar"]["recall_minimum"])
        phase = "export"
        checkpoint = output_dir / "marker-center-geometry-finetune-v15-p1.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.export_contract(), "selected_threshold": selected["threshold"], "source_checkpoint_sha256": config["source_checkpoint_sha256"]}, checkpoint)
        onnx_path = output_dir / "marker-center-geometry-finetune-v15-p1.onnx"
        example = torch.zeros((8, 3, 33, 33), dtype=torch.float32)
        torch.onnx.export(model, example, onnx_path, input_names=["candidate_patches"], output_names=["candidate_predictions"], dynamic_axes={"candidate_patches": {0: "candidate_count"}, "candidate_predictions": {0: "candidate_count"}}, opset_version=18, dynamo=False)
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        phase = "parity"
        parity_rows = []
        for candidate_count in (1, 8, 17):
            parity_input = torch.zeros((candidate_count, 3, 33, 33), dtype=torch.float32)
            with torch.inference_mode():
                expected = model(parity_input).numpy()
            actual = session.run(["candidate_predictions"], {"candidate_patches": parity_input.numpy()})[0]
            parity_rows.append({"candidate_count": candidate_count, "maximum_absolute_error": float(np.max(np.abs(expected - actual)))})
        parity = max(float(row["maximum_absolute_error"]) for row in parity_rows)
        report = {"schema": "graphreader.marker-center-geometry-finetune-candidate.v15", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "dev_passed" if dev_passed and parity <= float(config["onnx_parity_tolerance"]) else "failed_dev", "synthetic_only": True, "private_data": False, "private_or_article_images": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "sealed_runs": 0, "training_authorization": authorization.binding, "source_checkpoint_sha256": config["source_checkpoint_sha256"], "input_proposal_manifest_sha256": config["input_proposal_manifest_sha256"], "training_example_count": len(labels), "hard_negative_example_count": int(hard.sum()), "dev_comparisons": comparisons, "selected": selected, "dev_gate_passed": dev_passed, "checkpoint_sha256": sha256_file(checkpoint), "onnx_sha256": sha256_file(onnx_path), "onnx_provider": "CPUExecutionProvider", "onnx_parity_maximum_absolute_error": parity, "onnx_parity_tolerance": config["onnx_parity_tolerance"], "onnx_parity_passed": parity <= float(config["onnx_parity_tolerance"]), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "production_approval": False, "release_eligible": False}
        report["dataset_manifest_path"] = manifest_path.relative_to(REPO_ROOT).as_posix()
        report["optimizer_steps"] = optimizer_steps
        report["onnx_parity_rows"] = parity_rows
        report["model_license"] = config["model_license"]
    except Exception as error:
        report = {"schema": "graphreader.marker-center-geometry-finetune-failure.v15", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner", "phase": phase, "exception_type": type(error).__name__, "exception_message": str(error), "synthetic_only": True, "private_data": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "sealed_runs": 0, "training_authorization": authorization.binding, "completed_utc": datetime.now(timezone.utc).isoformat()}
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
