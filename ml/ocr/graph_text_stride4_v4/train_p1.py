# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for the stride-4 detector defect class."""

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
from ml.ocr.graph_text_ignore_band_v3.train_p1 import (
    _configure,
    _export,
    _selection_passed,
    evaluate_frames,
    normalize_bgr,
)
from ml.ocr.graph_text_ignore_band_v3.train_p3 import _p3_loss

from .dataset import build_training_arrays, build_validation_split, split_fingerprint, training_split_fingerprint
from .model import Stride4TextRegionNet
from .protocol import BATCH_SIZE, REVISION, SEED, TASK, TRAIN_SAMPLE_COUNT


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/ocr/graph_text_stride4_v4/training/p1.json")
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_stride4_v4/artifacts/P1-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_stride4_v4/dataset.py"),
    Path("ml/ocr/graph_text_stride4_v4/model.py"),
    Path("ml/ocr/graph_text_stride4_v4/protocol.py"),
    Path("ml/ocr/graph_text_stride4_v4/train_p1.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/dataset.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p1.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p3.py"),
    Path("ml/ocr/official_bakeoff/structure_consensus_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
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
            raise RuntimeError("Stride-4 trigger result changed")
        selection_path = REPO_ROOT / str(config["selection_manifest_path"])
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Stride-4 selection manifest changed")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if training_split_fingerprint() != selection["training_split_fingerprint"]:
            raise RuntimeError("Stride-4 frozen training split changed")
        validation_frames = build_validation_split()
        if split_fingerprint(validation_frames) != selection["validation_split_fingerprint"]:
            raise RuntimeError("Stride-4 frozen validation split changed")
        seal_path = REPO_ROOT / str(config["sealed_public_test_seal_path"])
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Stride-4 sealed-public seal changed")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / str(seal["fixture_archive_path"])) != seal["fixture_archive_sha256"]:
            raise RuntimeError("Stride-4 sealed-public archive changed")
        if config["training_sample_count"] != TRAIN_SAMPLE_COUNT:
            raise RuntimeError("Stride-4 training sample count changed")
        expected_steps = int(config["epochs"]) * (TRAIN_SAMPLE_COUNT // BATCH_SIZE)
        if config["expected_optimizer_steps"] != expected_steps:
            raise RuntimeError("Stride-4 optimizer-step budget changed")
        ceiling = float(config["boundary_probability_ceiling"])
        margin_weight = float(config["boundary_margin_loss_weight"])
        if ceiling != 0.25 or margin_weight != 1.0:
            raise RuntimeError("Stride-4 ignored-boundary objective changed")

        generator = _configure(int(config["seed"]))
        bgr, targets, supervision_masks = build_training_arrays()
        bgr_tensor = torch.from_numpy(bgr)
        target_tensor = torch.from_numpy(targets).float() / 255.0
        supervision_tensor = torch.from_numpy(supervision_masks).float() / 255.0
        model = Stride4TextRegionNet(seed=int(config["seed"]))
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
            raise RuntimeError("Stride-4 ONNX preflight violated the probability contract")
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
                supervision = supervision_tensor.index_select(0, indices)
                total, binary, dice, margin = _p3_loss(
                    model(inputs),
                    target,
                    supervision,
                    boundary_probability_ceiling=ceiling,
                    boundary_margin_loss_weight=margin_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer_steps += 1
                losses.append((float(total.detach()), float(binary.detach()), float(dice.detach()), float(margin.detach())))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({
                    "epoch": epoch + 1,
                    "total": sum(item[0] for item in losses) / len(losses),
                    "masked_ohem_binary_cross_entropy": sum(item[1] for item in losses) / len(losses),
                    "masked_dice": sum(item[2] for item in losses) / len(losses),
                    "ignored_boundary_margin": sum(item[3] for item in losses) / len(losses),
                })
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("Stride-4 optimizer-step count differs from preregistration")

        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-stride4-v4-p1.pt"
        torch.save({"state_dict": model.state_dict(), "revision": REVISION, "candidate_id": CANDIDATE_ID}, checkpoint_path)
        onnx_path = output_dir / "graph-text-stride4-v4-p1.onnx"
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
        probability_contract_passed = np.isfinite(actual).all() and float(actual.min()) >= 0.0 and float(actual.max()) <= 1.0

        phase = "selection"
        validation_started = time.perf_counter()
        metrics = evaluate_frames(validation_frames, lambda tensor: session.run([output_name], {input_name: tensor})[0])
        selection_passed = _selection_passed(metrics)
        passed = selection_passed and parity_passed and probability_contract_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-graph-text-stride4-training-report.v1",
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
            "architecture": config["architecture"],
            "maximum_downsampling_factor": config["maximum_downsampling_factor"],
            "training_sample_count": len(bgr_tensor),
            "training_source_count": config["training_source_count"],
            "tiles_per_source": config["tiles_per_source"],
            "boundary_probability_ceiling": ceiling,
            "boundary_margin_loss_weight": margin_weight,
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
            "schema": "graphreader.ocr-graph-text-stride4-training-failure.v1",
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
        "selection_metrics": {key: value for key, value in report["selection_metrics"].items() if key != "records"},
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
