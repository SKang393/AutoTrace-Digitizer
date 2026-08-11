# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for the ignore-band graph text detector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Sequence

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff import structure_consensus_evaluate as detector_contract

from .dataset import (
    EvaluationFrame,
    build_training_arrays,
    build_validation_split,
    split_fingerprint,
    training_split_fingerprint,
)
from .model import IgnoreBandTextRegionNet
from .protocol import (
    BATCH_SIZE,
    CANONICAL_OUTPUT,
    CANDIDATE_ID,
    DICE_LOSS_WEIGHT,
    EMPTY_TARGET_NEGATIVE_PIXELS,
    EPOCHS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LEARNING_RATE,
    MATCH_IOU_MINIMUM,
    MINIMUM_NEGATIVE_PIXELS,
    NEGATIVE_TO_POSITIVE_RATIO,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    REVISION,
    SEED,
    TASK,
    WEIGHT_DECAY,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("ml/ocr/graph_text_ignore_band_v3/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_ignore_band_v3/dataset.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/model.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/protocol.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p1.py"),
    Path("ml/ocr/official_bakeoff/structure_consensus_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    return torch.Generator().manual_seed(seed)


def normalize_bgr(values: torch.Tensor) -> torch.Tensor:
    means = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    scales = torch.tensor((1 / 0.229, 1 / 0.224, 1 / 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    return ((values.float().permute(0, 3, 1, 2) / 255.0) - means) * scales


def _loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    supervision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.shape != target.shape or target.shape != supervision_mask.shape:
        raise ValueError("Ignore-band loss tensors must have identical shapes")
    epsilon = torch.finfo(probabilities.dtype).eps
    bounded = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    valid = supervision_mask > 0.5
    positive = (target > 0.5) & valid
    negative = (target <= 0.5) & valid
    sample_binary: list[torch.Tensor] = []
    for sample_index in range(probabilities.shape[0]):
        positive_loss = -torch.log(bounded[sample_index][positive[sample_index]])
        negative_loss = -torch.log(1.0 - bounded[sample_index][negative[sample_index]])
        if positive_loss.numel() > 0:
            negative_count = min(
                negative_loss.numel(),
                max(MINIMUM_NEGATIVE_PIXELS, positive_loss.numel() * NEGATIVE_TO_POSITIVE_RATIO),
            )
        else:
            negative_count = min(negative_loss.numel(), EMPTY_TARGET_NEGATIVE_PIXELS)
        selected_negative = (
            torch.topk(negative_loss, k=negative_count, largest=True, sorted=False).values
            if negative_count > 0
            else negative_loss
        )
        numerator = positive_loss.sum() + selected_negative.sum()
        denominator = max(1, positive_loss.numel() + selected_negative.numel())
        sample_binary.append(numerator / denominator)
    binary = torch.stack(sample_binary).mean()
    valid_float = valid.to(dtype=probabilities.dtype)
    masked_probability = probabilities * valid_float
    masked_target = target * valid_float
    intersection = (masked_probability * masked_target).sum(dim=(1, 2, 3))
    denominator = masked_probability.sum(dim=(1, 2, 3)) + masked_target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return binary + (DICE_LOSS_WEIGHT * dice), binary, dice


def _export(model: nn.Module, path: Path) -> None:
    example = torch.zeros((1, 3, PATCH_HEIGHT, PATCH_WIDTH), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["image"],
        output_names=["probabilities"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "probabilities": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=18,
        dynamo=False,
    )


def _iou(region: Any, truth: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = truth
    intersection_width = max(0.0, min(region.bounds.right, right) - max(region.bounds.left, left))
    intersection_height = max(0.0, min(region.bounds.bottom, bottom) - max(region.bounds.top, top))
    intersection = intersection_width * intersection_height
    union = region.bounds.width * region.bounds.height + (right - left) * (bottom - top) - intersection
    return 0.0 if union <= 0 else intersection / union


def evaluate_frames(
    frames: Sequence[EvaluationFrame],
    runner: Callable[[np.ndarray], np.ndarray],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    exact_count = false_regions = duplicate_regions = exclusion_false = 0
    text_count = text_exact = text_missed = text_false = text_multi = 0
    exclusion_count = exclusion_exact = 0
    for frame in frames:
        tensor = detector_contract.detector_tensor(frame.detector_bgr, FRAME_WIDTH, FRAME_HEIGHT)
        output = np.asarray(runner(tensor), dtype=np.float32)
        expected_shape = (1, 1, int(tensor.shape[2]), int(tensor.shape[3]))
        if output.shape != expected_shape or not np.isfinite(output).all():
            raise RuntimeError(f"Ignore-band runner returned invalid output for {frame.case_id}: {output.shape}")
        if float(output.min()) < 0.0 or float(output.max()) > 1.0:
            raise RuntimeError(f"Ignore-band runner returned non-probability output for {frame.case_id}")
        regions = detector_contract.db_model_regions(output, FRAME_WIDTH, FRAME_HEIGHT)
        truth = frame.truth_bbox
        ranked = [] if truth is None else sorted(
            ((_iou(region, truth), region) for region in regions),
            key=lambda item: (-item[0], item[1].region_id),
        )
        matched = int(bool(ranked) and ranked[0][0] >= MATCH_IOU_MINIMUM)
        expected = 1 if truth is not None else 0
        false = max(0, len(regions) - matched)
        duplicates = 0
        for left_index, left_region in enumerate(regions):
            for right_region in regions[left_index + 1 :]:
                if _iou(left_region, right_region.bounds.to_json()) >= 0.70:
                    duplicates += 1
        is_exact = matched == expected and false == 0 and duplicates == 0
        exact_count += int(is_exact)
        false_regions += false
        duplicate_regions += duplicates
        if truth is None:
            exclusion_count += 1
            exclusion_exact += int(is_exact)
            exclusion_false += len(regions)
        else:
            text_count += 1
            text_exact += int(is_exact)
            text_missed += int(matched == 0)
            text_false += false
            text_multi += int(len(regions) > 1)
        records.append({
            "case_id": frame.case_id,
            "kind": frame.kind,
            "source_sha256": frame.source_sha256,
            "detector_bgr_sha256": frame.detector_bgr_sha256,
            "truth_bbox": list(truth) if truth is not None else None,
            "prediction_count": len(regions),
            "matched_region_count": matched,
            "false_region_count": false,
            "duplicate_region_count": duplicates,
            "best_truth_iou": ranked[0][0] if ranked else 0.0,
            "exact": is_exact,
        })
    return {
        "fixture_count": len(frames),
        "exact_fixture_count": exact_count,
        "exact_rate": exact_count / max(1, len(frames)),
        "false_region_count": false_regions,
        "duplicate_region_count": duplicate_regions,
        "exclusion_false_region_count": exclusion_false,
        "text_fixture_count": text_count,
        "text_exact_fixture_count": text_exact,
        "text_missed_fixture_count": text_missed,
        "text_false_region_count": text_false,
        "text_multi_region_fixture_count": text_multi,
        "exclusion_fixture_count": exclusion_count,
        "exclusion_exact_fixture_count": exclusion_exact,
        "records": records,
    }


def _selection_passed(metrics: dict[str, object]) -> bool:
    return (
        int(metrics["exact_fixture_count"]) == int(metrics["fixture_count"])
        and int(metrics["false_region_count"]) == 0
        and int(metrics["duplicate_region_count"]) == 0
        and int(metrics["exclusion_false_region_count"]) == 0
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
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "candidate-report.json"
    phase = "initialization"
    optimizer_steps = 0
    started = time.perf_counter()
    try:
        if sha256_file(REPO_ROOT / config["trigger_result_path"]) != config["trigger_result_sha256"]:
            raise RuntimeError("Ignore-band trigger evidence changed")
        selection_path = REPO_ROOT / str(config["selection_manifest_path"])
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Ignore-band selection manifest changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if training_split_fingerprint() != selection["train_split_fingerprint"]:
            raise RuntimeError("Ignore-band training split changed")
        validation_frames = build_validation_split()
        if split_fingerprint(validation_frames) != selection["validation_split_fingerprint"]:
            raise RuntimeError("Ignore-band validation split changed")
        seal_path = REPO_ROOT / str(config["sealed_public_test_seal_path"])
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Ignore-band sealed-public seal changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / str(seal["fixture_archive_path"])) != seal["fixture_archive_sha256"]:
            raise RuntimeError("Ignore-band sealed-public archive changed")
        expected_constants = {
            "negative_to_positive_ratio": NEGATIVE_TO_POSITIVE_RATIO,
            "minimum_negative_pixels": MINIMUM_NEGATIVE_PIXELS,
            "empty_target_negative_pixels": EMPTY_TARGET_NEGATIVE_PIXELS,
            "dice_loss_weight": DICE_LOSS_WEIGHT,
        }
        for key, expected in expected_constants.items():
            if config.get(key) != expected:
                raise RuntimeError(f"Ignore-band frozen training constant changed: {key}")

        generator = _configure(int(config["seed"]))
        bgr, targets, supervision_masks = build_training_arrays()
        bgr_tensor = torch.from_numpy(bgr)
        target_tensor = torch.from_numpy(targets).float() / 255.0
        supervision_tensor = torch.from_numpy(supervision_masks).float() / 255.0
        model = IgnoreBandTextRegionNet(seed=int(config["seed"]))
        phase = "onnx_preflight"
        preflight = output_dir / "export-preflight.onnx"
        _export(model.eval(), preflight)
        preflight_session = ort.InferenceSession(str(preflight), providers=["CPUExecutionProvider"])
        preflight_output = np.asarray(
            preflight_session.run(
                [preflight_session.get_outputs()[0].name],
                {preflight_session.get_inputs()[0].name: normalize_bgr(bgr_tensor[:1]).numpy()},
            )[0],
            dtype=np.float32,
        )
        if not np.isfinite(preflight_output).all() or float(preflight_output.min()) < 0.0 or float(preflight_output.max()) > 1.0:
            raise RuntimeError("Ignore-band ONNX preflight violated the probability contract")
        preflight_sha256 = sha256_file(preflight)
        preflight.unlink()

        phase = "training"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(bgr_tensor), generator=generator)
            losses: list[tuple[float, float, float]] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                inputs = normalize_bgr(bgr_tensor.index_select(0, indices))
                target = target_tensor.index_select(0, indices)
                supervision = supervision_tensor.index_select(0, indices)
                total, binary, dice = _loss(model(inputs), target, supervision)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                losses.append((float(total.detach()), float(binary.detach()), float(dice.detach())))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({
                    "epoch": epoch + 1,
                    "total": sum(item[0] for item in losses) / len(losses),
                    "masked_ohem_binary_cross_entropy": sum(item[1] for item in losses) / len(losses),
                    "masked_dice": sum(item[2] for item in losses) / len(losses),
                })

        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-ignore-band-v3-p1.pt"
        torch.save({"state_dict": model.state_dict(), "revision": REVISION, "candidate_id": CANDIDATE_ID}, checkpoint_path)
        onnx_path = output_dir / "graph-text-ignore-band-v3-p1.onnx"
        _export(model, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        parity_input = normalize_bgr(bgr_tensor[:4]).numpy()
        with torch.inference_mode():
            expected = model(torch.from_numpy(parity_input)).numpy()
        actual = np.asarray(session.run([output_name], {input_name: parity_input})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])
        probability_contract_passed = (
            np.isfinite(actual).all() and float(actual.min()) >= 0.0 and float(actual.max()) <= 1.0
        )

        phase = "selection"
        validation_started = time.perf_counter()
        metrics = evaluate_frames(validation_frames, lambda tensor: session.run([output_name], {input_name: tensor})[0])
        selection_passed = _selection_passed(metrics)
        passed = selection_passed and parity_passed and probability_contract_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-graph-text-ignore-band-training-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "generalization_label_included": False,
            "training_authorization": authorization.binding,
            "training_sample_count": len(bgr_tensor),
            "validation_sample_count": len(validation_frames),
            "training_split_fingerprint": config["training_split_fingerprint"],
            "epochs": config["epochs"],
            "seed": config["seed"],
            "optimizer_steps": optimizer_steps,
            "loss_checkpoints": checkpoints,
            "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": session.get_providers()[0],
            "onnx_output_minimum": float(actual.min()),
            "onnx_output_maximum": float(actual.max()),
            "probability_contract_passed": probability_contract_passed,
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "onnx_preflight_sha256": preflight_sha256,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "validation_elapsed_ms": round((time.perf_counter() - validation_started) * 1000.0, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-graph-text-ignore-band-training-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "production_approval": False,
            "release_eligible": False,
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
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
    print(json.dumps({
        "candidate_id": report["candidate_id"],
        "status": report["status"],
        "selection_gate_passed": report["selection_gate_passed"],
        "probability_contract_passed": report["probability_contract_passed"],
        "onnx_parity_passed": report["onnx_parity_passed"],
        "selection_metrics": {
            key: value for key, value in report["selection_metrics"].items() if key != "records"
        },
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
