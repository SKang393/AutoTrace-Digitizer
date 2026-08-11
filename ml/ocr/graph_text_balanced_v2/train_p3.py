# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use exclusion-only hard-negative P3 training."""

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

from .dataset import (
    build_training_arrays,
    build_validation_split,
    split_fingerprint,
    training_split_fingerprint,
)
from .model import BalancedRecallTextRegionNet
from .protocol import (
    DICE_LOSS_WEIGHT,
    PATCH_HEIGHT,
    PATCH_WIDTH,
    POSITIVE_BCE_WEIGHT,
    REVISION,
    TASK,
)
from .train_p1 import (
    REPO_ROOT,
    _configure,
    _export,
    _selection_passed,
    evaluate_frames,
    normalize_bgr,
)
from .train_p2 import HARD_NEGATIVE_LOSS_WEIGHT, HARD_NEGATIVE_TOPK_FRACTION


CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/ocr/graph_text_balanced_v2/training/p3.json")
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_balanced_v2/artifacts/P3-run")
HARD_NEGATIVE_SCOPE = "empty_target_exclusion_patches_only"
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_balanced_v2/dataset.py"),
    Path("ml/ocr/graph_text_balanced_v2/model.py"),
    Path("ml/ocr/graph_text_balanced_v2/protocol.py"),
    Path("ml/ocr/graph_text_balanced_v2/train_p1.py"),
    Path("ml/ocr/graph_text_balanced_v2/train_p2.py"),
    Path("ml/ocr/graph_text_balanced_v2/train_p3.py"),
    Path("ml/ocr/official_bakeoff/structure_consensus_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = torch.finfo(probabilities.dtype).eps
    bounded = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
    weighted_binary = -(
        (POSITIVE_BCE_WEIGHT * target * torch.log(bounded))
        + ((1.0 - target) * torch.log(1.0 - bounded))
    ).mean()
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    background_loss = -torch.log(1.0 - bounded) * (1.0 - target)
    flattened = background_loss.flatten(start_dim=1)
    topk_count = max(1, int(flattened.shape[1] * HARD_NEGATIVE_TOPK_FRACTION))
    per_sample = torch.topk(
        flattened,
        k=topk_count,
        dim=1,
        largest=True,
        sorted=False,
    ).values.mean(dim=1)
    empty_target = target.flatten(start_dim=1).sum(dim=1).eq(0)
    empty_count = empty_target.to(dtype=per_sample.dtype).sum()
    if bool(empty_target.any()):
        hard_negative = (
            per_sample * empty_target.to(dtype=per_sample.dtype)
        ).sum() / empty_count
    else:
        hard_negative = per_sample.sum() * 0.0
    total = (
        weighted_binary
        + (DICE_LOSS_WEIGHT * dice)
        + (HARD_NEGATIVE_LOSS_WEIGHT * hard_negative)
    )
    return total, weighted_binary, dice, hard_negative


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
            raise RuntimeError("Balanced-recall P3 trigger evidence changed")
        selection_path = REPO_ROOT / str(config["selection_manifest_path"])
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Balanced-recall P3 selection manifest changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if training_split_fingerprint() != selection["train_split_fingerprint"]:
            raise RuntimeError("Balanced-recall P3 training split changed")
        validation_frames = build_validation_split()
        if split_fingerprint(validation_frames) != selection["validation_split_fingerprint"]:
            raise RuntimeError("Balanced-recall P3 validation split changed")
        seal_path = REPO_ROOT / str(config["sealed_public_test_seal_path"])
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Balanced-recall P3 sealed-public seal changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / str(seal["fixture_archive_path"])) != seal["fixture_archive_sha256"]:
            raise RuntimeError("Balanced-recall P3 sealed-public archive changed")
        if float(config["hard_negative_topk_fraction"]) != HARD_NEGATIVE_TOPK_FRACTION:
            raise RuntimeError("Balanced-recall P3 hard-negative fraction changed")
        if float(config["hard_negative_loss_weight"]) != HARD_NEGATIVE_LOSS_WEIGHT:
            raise RuntimeError("Balanced-recall P3 hard-negative weight changed")
        if config["hard_negative_scope"] != HARD_NEGATIVE_SCOPE:
            raise RuntimeError("Balanced-recall P3 hard-negative scope changed")

        generator = _configure(int(config["seed"]))
        bgr, targets = build_training_arrays()
        bgr_tensor = torch.from_numpy(bgr)
        target_tensor = torch.from_numpy(targets).float() / 255.0
        model = BalancedRecallTextRegionNet(seed=int(config["seed"]))
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
        if (
            not np.isfinite(preflight_output).all()
            or float(preflight_output.min()) < 0.0
            or float(preflight_output.max()) > 1.0
        ):
            raise RuntimeError("Balanced-recall P3 ONNX preflight violated the probability contract")
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
            losses: list[tuple[float, float, float, float]] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start : start + int(config["batch_size"])]
                inputs = normalize_bgr(bgr_tensor.index_select(0, indices))
                target = target_tensor.index_select(0, indices)
                total, binary, dice, hard_negative = _loss(model(inputs), target)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                losses.append(
                    (
                        float(total.detach()),
                        float(binary.detach()),
                        float(dice.detach()),
                        float(hard_negative.detach()),
                    )
                )
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append(
                    {
                        "epoch": epoch + 1,
                        "total": sum(item[0] for item in losses) / len(losses),
                        "weighted_binary_cross_entropy": sum(item[1] for item in losses) / len(losses),
                        "dice": sum(item[2] for item in losses) / len(losses),
                        "hard_negative_background": sum(item[3] for item in losses) / len(losses),
                    }
                )

        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-balanced-recall-v2-p3.pt"
        torch.save(
            {"state_dict": model.state_dict(), "revision": REVISION, "candidate_id": CANDIDATE_ID},
            checkpoint_path,
        )
        onnx_path = output_dir / "graph-text-balanced-recall-v2-p3.onnx"
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
        metrics = evaluate_frames(
            validation_frames,
            lambda tensor: session.run([output_name], {input_name: tensor})[0],
        )
        selection_passed = _selection_passed(metrics)
        passed = selection_passed and parity_passed and probability_contract_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-graph-text-balanced-recall-training-report.v1",
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
            "positive_bce_weight": POSITIVE_BCE_WEIGHT,
            "dice_loss_weight": DICE_LOSS_WEIGHT,
            "hard_negative_topk_fraction": HARD_NEGATIVE_TOPK_FRACTION,
            "hard_negative_loss_weight": HARD_NEGATIVE_LOSS_WEIGHT,
            "hard_negative_scope": HARD_NEGATIVE_SCOPE,
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
        complete_training_candidate(
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-graph-text-balanced-recall-training-failure.v1",
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
        complete_training_candidate(
            authorization,
            status="failed_runner",
            report_sha256=sha256_file(report_path),
        )
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
                "selection_gate_passed": report["selection_gate_passed"],
                "probability_contract_passed": report["probability_contract_passed"],
                "onnx_parity_passed": report["onnx_parity_passed"],
                "selection_metrics": {
                    key: value
                    for key, value in report["selection_metrics"].items()
                    if key != "records"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
