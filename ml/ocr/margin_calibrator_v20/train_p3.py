# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 complete-stream multitask training for OCR V20."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_split_archive, proposal_summary, split_fingerprint
from .multitask_model import CompleteStreamMultitaskCalibrator
from .pipeline import ProposalRecord, evaluate_thresholds, extract_features, select_robust_window
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    ROLE_ORDER,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/margin_calibrator_v20")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "multitask_model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p3.py",
    P2_RESULT_PATH,
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
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


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V20 P3 requires CPUExecutionProvider only")
    return session


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model,
        example,
        path,
        input_names=["proposal_evidence"],
        output_names=["calibration_logits"],
        dynamic_axes={
            "proposal_evidence": {0: "proposal_count"},
            "calibration_logits": {0: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )


def _role_targets(
    scenes: tuple[object, ...],
    records: tuple[ProposalRecord, ...],
) -> np.ndarray:
    role_indices = {role: index for index, role in enumerate(ROLE_ORDER)}
    targets: list[int] = []
    for record in records:
        if record.truth_index < 0:
            targets.append(-100)
            continue
        scene = scenes[record.scene_index]
        role = scene.truths[record.truth_index].role
        targets.append(role_indices[role])
    return np.asarray(targets, dtype=np.int64)


def _calibrated_records(
    records: tuple[ProposalRecord, ...],
    output: np.ndarray,
) -> tuple[ProposalRecord, ...]:
    if output.shape != (len(records), 2 + len(ROLE_ORDER)):
        raise RuntimeError("OCR V20 P3 multitask output contract changed")
    role_indices = output[:, 2:].argmax(axis=1)
    return tuple(
        ProposalRecord(
            record.scene_index,
            record.candidate_index,
            record.truth_index,
            record.predicted_text,
            ROLE_ORDER[int(role_index)],
        )
        for record, role_index in zip(records, role_indices, strict=True)
    )


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V20 P3 output exists: {output_dir}")
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
    phase = "preflight"
    optimizer_steps = 0
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(
            REPO_ROOT, RUNNER_SOURCE_PATHS
        ):
            raise RuntimeError("OCR V20 P3 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V20 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V20 public seal changed")
        if (
            config["trigger_result_path"] != P2_RESULT_PATH.as_posix()
            or config["trigger_result_sha256"] != sha256_file(REPO_ROOT / P2_RESULT_PATH)
        ):
            raise RuntimeError("OCR V20 aggregate P2 result changed")
        p2_result = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
        if (
            p2_result["status"] != "failed_selection"
            or p2_result["case_level_details_emitted"] is not False
            or p2_result["public_gate_archive_opened"] is not False
            or p2_result["public_gate_evaluations"] != 0
            or p2_result["selection_metrics"]["false_positives"] != 4
            or p2_result["selection_metrics"]["false_negatives"] != 4
            or p2_result["selection_metrics"]["minimum_role_accuracy"] != 0.5625
            or p2_result["passing_threshold_window"] != []
        ):
            raise RuntimeError("OCR V20 P2 trigger is not the consumed aggregate-only failure")
        for relative, expected in {
            DETECTOR_PATH: DETECTOR_SHA256,
            RECOGNIZER_PATH: RECOGNIZER_SHA256,
            RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        }.items():
            if sha256_file(REPO_ROOT / relative) != expected:
                raise RuntimeError(f"OCR V20 P3 exact frozen input changed: {relative}")

        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        splits: dict[str, tuple[object, ...]] = {}
        for split in ("train", "validation"):
            registered = selection[split]
            archive = REPO_ROOT / registered["fixture_archive_path"]
            manifest = REPO_ROOT / registered["private_manifest_path"]
            if (
                sha256_file(archive) != registered["fixture_archive_sha256"]
                or sha256_file(manifest) != registered["private_manifest_sha256"]
            ):
                raise RuntimeError(f"OCR V20 stored {split} fixture bytes changed")
            scenes = load_split_archive(archive, manifest, expected_split=split)
            summary = proposal_summary(scenes)
            if (
                split_fingerprint(scenes) != registered["split_fingerprint"]
                or any(summary[key] != registered[key] for key in summary)
            ):
                raise RuntimeError(f"OCR V20 stored {split} fixtures violate the frozen split")
            splits[split] = scenes

        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

        def detector_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                detector_session.run(None, {detector_input: np.ascontiguousarray(values)})[0],
                dtype=np.float32,
            )

        def recognizer_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(
                recognizer_session.run(None, {recognizer_input: np.ascontiguousarray(values)})[0],
                dtype=np.float32,
            )

        phase = "complete_training_feature_execution"
        train_values, train_labels, train_records, training_evidence = extract_features(
            splits["train"],
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["training_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        for key in (
            "scene_count",
            "proposal_count",
            "positive_proposal_count",
            "negative_proposal_count",
        ):
            if training_evidence[key] != config["training_counts"][key]:
                raise RuntimeError(f"OCR V20 P3 complete training count changed: {key}")
        role_targets_array = _role_targets(splits["train"], train_records)
        role_counts = {
            role: int(np.sum(role_targets_array == index))
            for index, role in enumerate(ROLE_ORDER)
        }
        if role_counts != config["training_role_counts"]:
            raise RuntimeError("OCR V20 P3 training role counts changed")
        training_evidence["role_target_counts"] = role_counts

        generator = _configure(int(config["seed"]))
        model = CompleteStreamMultitaskCalibrator(seed=int(config["seed"]))
        tensors = torch.from_numpy(train_values)
        proposal_targets = torch.from_numpy(train_labels)
        role_targets = torch.from_numpy(role_targets_array)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        proposal_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(
                (float(config["negative_class_weight"]), float(config["positive_class_weight"]))
            )
        )
        role_criterion = nn.CrossEntropyLoss()
        loss_checkpoints: list[dict[str, float | int]] = []
        phase = "multitask_training"
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(tensors), generator=generator)
            losses: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                batch_proposal_targets = proposal_targets.index_select(0, indices)
                batch_role_targets = role_targets.index_select(0, indices)
                batch_output = model(tensors.index_select(0, indices))
                proposal_logits = batch_output[:, :2]
                differences = proposal_logits[:, 1] - proposal_logits[:, 0]
                positive = differences[batch_proposal_targets == 1]
                negative = differences[batch_proposal_targets == 0]
                positive_loss = (
                    torch.relu(positive.new_tensor(float(config["positive_logit_margin"])) - positive)
                    .square()
                    .mean()
                    if positive.numel()
                    else differences.new_zeros(())
                )
                negative_loss = (
                    torch.relu(negative - negative.new_tensor(float(config["negative_logit_margin"])))
                    .square()
                    .mean()
                    if negative.numel()
                    else differences.new_zeros(())
                )
                pairwise_loss = (
                    torch.relu(
                        differences.new_tensor(float(config["pairwise_logit_margin"]))
                        - (positive[:, None] - negative[None, :])
                    )
                    .square()
                    .mean()
                    if positive.numel() and negative.numel()
                    else differences.new_zeros(())
                )
                role_mask = batch_role_targets >= 0
                calibrated_role_loss = (
                    role_criterion(batch_output[role_mask, 2:], batch_role_targets[role_mask])
                    if bool(role_mask.any())
                    else differences.new_zeros(())
                )
                loss = (
                    proposal_criterion(proposal_logits, batch_proposal_targets)
                    + float(config["margin_loss_weight"]) * (positive_loss + negative_loss)
                    + float(config["pairwise_loss_weight"]) * pairwise_loss
                    + float(config["role_loss_weight"]) * calibrated_role_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                loss_checkpoints.append(
                    {"epoch": epoch + 1, "loss": sum(losses) / len(losses)}
                )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V20 P3 optimizer-step count changed")

        phase = "export"
        checkpoint_path = output_dir / "graph-text-margin-calibrator-v20-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-margin-calibrator-v20-p3.onnx"
        parity_values = tensors[:256]
        model.eval()
        _export(model, parity_values, onnx_path)
        calibrator_session = _cpu_session(onnx_path)
        calibrator_input = calibrator_session.get_inputs()[0].name
        with torch.inference_mode():
            expected_output = model(parity_values).numpy()
        actual_output = np.asarray(
            calibrator_session.run(
                None,
                {calibrator_input: np.ascontiguousarray(parity_values.numpy())},
            )[0],
            dtype=np.float32,
        )
        parity_error = float(np.max(np.abs(expected_output - actual_output)))
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR

        phase = "complete_visible_selection"
        validation_values, _, validation_records, selection_evidence = extract_features(
            splits["validation"],
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["validation_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        for key in (
            "scene_count",
            "proposal_count",
            "positive_proposal_count",
            "negative_proposal_count",
        ):
            if selection_evidence[key] != config["validation_counts"][key]:
                raise RuntimeError(f"OCR V20 P3 complete validation count changed: {key}")
        validation_output = np.asarray(
            calibrator_session.run(
                None,
                {calibrator_input: np.ascontiguousarray(validation_values)},
            )[0],
            dtype=np.float32,
        )
        calibrated_records = _calibrated_records(validation_records, validation_output)
        selection_evidence["calibrator_input_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_values).tobytes(order="C")
        ).hexdigest()
        selection_evidence["calibrator_output_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_output).tobytes(order="C")
        ).hexdigest()
        selection_evidence["calibrator_onnx_sha256"] = sha256_file(onnx_path)
        selection_evidence["detector_prefilter_applied"] = False
        selection_evidence["calibrated_role_head_applied"] = True
        comparisons = evaluate_thresholds(
            splits["validation"],
            calibrated_records,
            validation_output[:, :2],
            tuple(float(value) for value in config["selection_thresholds"]),
            selection_evidence,
        )
        robust = select_robust_window(comparisons)
        selected = robust[0] if robust else max(
            comparisons,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                -item["metrics"]["false_negatives"],
                item["metrics"]["recognition_exact"],
                item["metrics"]["role_accuracy"],
            ),
        )
        window = robust[1] if robust else ()
        passed = robust is not None and parity_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-margin-calibrator-candidate.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "p2_case_details_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p2_aggregate_metrics_only_used_for_design": True,
            "isolated_change": config["isolated_change"],
            "architecture": config["architecture"],
            "optimizer_steps": optimizer_steps,
            "training_evidence": training_evidence,
            "training_role_counts": role_counts,
            "loss_checkpoints": loss_checkpoints,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "training_authorization": authorization.binding,
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
            "schema": "graphreader.ocr-margin-calibrator-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "public_gate_archive_opened": False,
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
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_steps": report["optimizer_steps"],
                "selected_threshold": report["selected_threshold"],
                "passing_threshold_window": report["passing_threshold_window"],
                "selection_metrics": report["selection_metrics"],
                "onnx_parity_maximum_absolute_error": report[
                    "onnx_parity_maximum_absolute_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
