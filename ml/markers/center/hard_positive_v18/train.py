# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authorization-ready V18 train-only hard-positive mining runner."""

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

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.proposal_geometry_v13.dataset import build_selection_scenes, seal_manifest
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals
from ml.markers.center.scale_classifier_v16.model import ModelConfig, ScaleClassifierNet
from ml.markers.center.scale_classifier_v16.train import _configure, _evaluate, _loss
from ml.markers.center.metric_aligned_v17.train import _examples as v17_examples
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .protocol import HARD_POSITIVE_REPEAT_COUNT, HARD_POSITIVE_THRESHOLD, V13_MANIFEST_SHA256, V17_RESULT_PATH, V17_RESULT_SHA256, WARMUP_EPOCHS, FINISH_EPOCHS

REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-hard-positive-v18"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/hard_positive_v18/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/hard_positive_v18/protocol.py"),
    Path("ml/markers/center/hard_positive_v18/train.py"),
    Path("ml/markers/center/metric_aligned_v17/P1_RESULT.json"),
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


def _positive_examples(scenes):
    patches, labels, offsets, radii, hard = [], [], [], [], []
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        coordinates = proposals.coordinates
        centers = torch.tensor(scene.centers, dtype=torch.float32)
        scene_radii = torch.tensor(scene.radii, dtype=torch.float32)
        distances = torch.cdist(coordinates, centers)
        nearest, nearest_index = distances.min(dim=1)
        labels_scene = nearest.le(3.0)
        positive = torch.nonzero(labels_scene, as_tuple=False).flatten()
        patches.append(proposals.patches.index_select(0, positive))
        labels.append(torch.ones(len(positive), dtype=torch.float32))
        offsets.append((centers.index_select(0, nearest_index) - coordinates).index_select(0, positive) / 4.0)
        radii.append(scene_radii.index_select(0, nearest_index).index_select(0, positive))
        hard.append(torch.zeros(len(positive), dtype=torch.bool))
    return torch.cat(patches), torch.cat(labels), torch.cat(offsets), torch.cat(radii), torch.cat(hard)


def _mine_hard_positives(scenes, model, *, threshold: float, repeat_count: int):
    patches, labels, offsets, radii, hard = [], [], [], [], []
    mined = 0
    by_scale = {"small": 0, "medium": 0, "large": 0}
    model.eval()
    for scene in scenes:
        proposals = filter_proposals(scene.tensor, extract_proposals(scene.tensor))
        coordinates = proposals.coordinates
        centers = torch.tensor(scene.centers, dtype=torch.float32)
        scene_radii = torch.tensor(scene.radii, dtype=torch.float32)
        distances = torch.cdist(coordinates, centers)
        nearest, nearest_index = distances.min(dim=1)
        positive = torch.nonzero(nearest.le(3.0), as_tuple=False).flatten()
        if len(positive) == 0:
            continue
        with torch.inference_mode():
            scores = model(proposals.patches.index_select(0, positive))[:, 0]
        low = positive[scores < threshold]
        if len(low):
            patches.append(proposals.patches.index_select(0, low).repeat((repeat_count, 1, 1, 1)))
            labels.append(torch.ones(len(low) * repeat_count, dtype=torch.float32))
            offsets.append((centers.index_select(0, nearest_index).index_select(0, low) - coordinates.index_select(0, low)).repeat((repeat_count, 1)) / 4.0)
            radii.append(scene_radii.index_select(0, nearest_index).index_select(0, low).repeat(repeat_count))
            hard.append(torch.zeros(len(low) * repeat_count, dtype=torch.bool))
            mined += len(low)
            for radius in scene_radii.index_select(0, nearest_index).index_select(0, low).tolist():
                by_scale["small" if radius < 5 else "medium" if radius < 8 else "large"] += 1
    if not patches:
        return None, {"mined_positive_count": 0, "repeated_example_count": 0, "by_scale": by_scale}
    return (torch.cat(patches), torch.cat(labels), torch.cat(offsets), torch.cat(radii), torch.cat(hard)), {"mined_positive_count": mined, "repeated_example_count": mined * repeat_count, "by_scale": by_scale}


def run(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID, config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS)
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started = time.perf_counter()
    try:
        previous = REPO_ROOT / V17_RESULT_PATH
        if sha256_file(previous) != V17_RESULT_SHA256 or json.loads(previous.read_text(encoding="utf-8")).get("dev_gate_passed") is not False:
            raise RuntimeError("V18 requires the failed V17 aggregate result")
        _configure(int(config["seed"]))
        manifest_path, manifest_sha256 = seal_manifest(output_dir)
        if manifest_sha256 != V13_MANIFEST_SHA256:
            raise RuntimeError("V13 proposal manifest changed")
        train_scenes = build_selection_scenes("train")
        generator = torch.Generator().manual_seed(int(config["seed"]) + 1)
        base = v17_examples(train_scenes, int(config["maximum_negative_per_positive"]), generator)
        model = ScaleClassifierNet(ModelConfig(seed=int(config["seed"])))
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        optimizer_steps = 0
        model.train()
        warmup_batches = (len(base[1]) + int(config["batch_size"]) - 1) // int(config["batch_size"])
        if len(base[1]) != int(config["expected_warmup_training_example_count"]):
            raise RuntimeError("V17 warmup example count changed")
        if warmup_batches * WARMUP_EPOCHS != int(config["expected_warmup_optimizer_steps"]):
            raise RuntimeError("V18 warmup optimizer-step count changed")
        for epoch in range(WARMUP_EPOCHS):
            order = torch.randperm(len(base[1]), generator=generator)
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                loss = _loss(model.forward_raw(base[0].index_select(0, indices)), base[1].index_select(0, indices), base[2].index_select(0, indices), base[3].index_select(0, indices), base[4].index_select(0, indices), positive_weight=float(config["positive_loss_weight"]), hard_weight=float(config["hard_negative_loss_weight"]))
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); optimizer_steps += 1
        phase = "train_only_mining"
        mined, mining_summary = _mine_hard_positives(train_scenes, model, threshold=HARD_POSITIVE_THRESHOLD, repeat_count=HARD_POSITIVE_REPEAT_COUNT)
        if mined is not None:
            final = tuple(torch.cat((base[index], mined[index])) for index in range(5))
        else:
            final = base
        model.train()
        for epoch in range(FINISH_EPOCHS):
            order = torch.randperm(len(final[1]), generator=generator)
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                loss = _loss(model.forward_raw(final[0].index_select(0, indices)), final[1].index_select(0, indices), final[2].index_select(0, indices), final[3].index_select(0, indices), final[4].index_select(0, indices), positive_weight=float(config["positive_loss_weight"]), hard_weight=float(config["hard_negative_loss_weight"]))
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); optimizer_steps += 1
        # Dev is first constructed or read only after all train-only work is complete.
        dev_scenes = build_selection_scenes("dev")
        model.eval()
        comparisons = [_evaluate(dev_scenes, model, float(threshold)) for threshold in config["selection_thresholds"]]
        selected = max(comparisons, key=lambda row: (min(float(row["precision"]), float(row["recall"])), float(row["f1"])))
        dev_passed = float(selected["precision"]) >= float(config["acceptance_bar"]["precision_minimum"]) and float(selected["recall"]) >= float(config["acceptance_bar"]["recall_minimum"])
        checkpoint = output_dir / "marker-center-hard-positive-v18-p1.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.export_contract(), "selected_threshold": selected["threshold"], "dataset_manifest_sha256": manifest_sha256, "hard_positive_threshold": HARD_POSITIVE_THRESHOLD, "hard_positive_repeat_count": HARD_POSITIVE_REPEAT_COUNT}, checkpoint)
        onnx_path = output_dir / "marker-center-hard-positive-v18-p1.onnx"
        example = torch.zeros((1, 3, 33, 33), dtype=torch.float32)
        torch.onnx.export(model, example, onnx_path, input_names=["candidate_patches"], output_names=["candidate_predictions"], dynamic_axes={"candidate_patches": {0: "candidate_count"}, "candidate_predictions": {0: "candidate_count"}}, opset_version=18, dynamo=False)
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("V18 parity requires CPUExecutionProvider")
        parity_rows = []
        for count in config["onnx_dynamic_candidate_counts"]:
            value = torch.zeros((int(count), 3, 33, 33), dtype=torch.float32)
            with torch.inference_mode():
                expected = model(value).numpy()
            actual = session.run(["candidate_predictions"], {"candidate_patches": value.numpy()})[0]
            parity_rows.append({"candidate_count": int(count), "maximum_absolute_error": float(np.max(np.abs(expected - actual)))})
        parity = max(float(row["maximum_absolute_error"]) for row in parity_rows)
        report = {"schema": "graphreader.marker-center-hard-positive-candidate.v18", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "dev_passed" if dev_passed and parity <= float(config["onnx_parity_tolerance"]) else "failed_dev", "synthetic_only": True, "private_data": False, "private_or_article_images": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "training_authorization": authorization.binding, "v17_result_path": V17_RESULT_PATH, "v17_result_sha256": V17_RESULT_SHA256, "model_license": config["model_license"], "dataset_manifest_sha256": manifest_sha256, "warmup_training_example_count": len(base[1]), "expected_warmup_optimizer_steps": int(config["expected_warmup_optimizer_steps"]), "mining": mining_summary, "final_training_example_count": len(final[1]), "optimizer_steps": optimizer_steps, "dev_comparisons": comparisons, "selected": selected, "dev_gate_passed": dev_passed, "checkpoint_sha256": sha256_file(checkpoint), "onnx_sha256": sha256_file(onnx_path), "onnx_provider": "CPUExecutionProvider", "onnx_dynamic_candidate_counts": parity_rows, "onnx_parity_maximum_absolute_error": parity, "onnx_parity_tolerance": config["onnx_parity_tolerance"], "onnx_parity_passed": parity <= float(config["onnx_parity_tolerance"]), "elapsed_ms": round((time.perf_counter() - started) * 1000, 3), "production_approval": False, "release_eligible": False}
    except Exception as error:
        report = {"schema": "graphreader.marker-center-hard-positive-failure.v18", "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner", "exception_type": type(error).__name__, "exception_message": str(error), "synthetic_only": True, "private_data": False, "public_gate_archive_opened": False, "public_gate_evaluations": 0, "training_authorization": authorization.binding, "completed_utc": datetime.now(timezone.utc).isoformat()}
        report_path.write_bytes(canonical_json_bytes(report)); complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path)); raise
    report_path.write_bytes(canonical_json_bytes(report)); complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path)); return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args(); report = run(args.output_dir.resolve()); print(json.dumps(report, indent=2, sort_keys=True)); return 0 if report["status"] == "dev_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
