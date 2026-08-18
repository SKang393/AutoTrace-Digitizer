# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use frozen-P2 monotonic veto-only repair for OCR V27 P3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate

from . import train_p2 as p2
from .model_p3 import FrozenP2MonotonicVetoProposalNet
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    OUTPUT_LOGIT_SCALE,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    STRUCTURE_FEATURE_COUNT,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/morphology_veto_proposal_v27")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_RESULT_SHA256 = "170be74dd092c030d07437d42e74dbe089892974cedfa38eceb66ccffd5409bb"
P2_CHECKPOINT_PATH = ROOT / "artifacts/P2-run/graph-text-morphology-veto-proposal-v27-p2.pt"
P2_CHECKPOINT_SHA256 = "5cc770cdf431d754d52598364d2c127390f65d85a52517a1ce0e0dd7e2e284f0"
P2_ONNX_PATH = ROOT / "artifacts/P2-run/graph-text-morphology-veto-proposal-v27-p2.onnx"
P2_ONNX_SHA256 = "bb57e22d61b7a7127eac09a7ddac716bc0456ae38e73d9fbaab283977fd190f5"
P2_REPORT_PATH = ROOT / "artifacts/P2-run/candidate-report.json"
P2_REPORT_SHA256 = "d1ccc39bcbe92a50773564059d8b9eb8521d1d0b8a13286c7b0f713dded33420"
RUNNER_SOURCE_PATHS = (
    *p2.RUNNER_SOURCE_PATHS,
    P2_RESULT_PATH,
    ROOT / "model_p3.py",
    ROOT / "train_p3.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _load_p2_state() -> dict[str, torch.Tensor]:
    payload = torch.load(
        REPO_ROOT / P2_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("OCR V27 P2 checkpoint state is missing")
    return state


def _objective(
    logits: torch.Tensor,
    p2_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    per_example = nn.functional.cross_entropy(
        logits, targets, weight=class_weights, reduction="none",
    )
    negative = targets == 0
    positive = targets == 1
    if not torch.any(positive) or not torch.any(negative):
        raise RuntimeError("OCR V27 P3 requires positive and negative proposals per scene")
    asymmetric = torch.where(
        negative,
        per_example * float(config["false_positive_weight"]),
        per_example,
    ).mean()
    margin = logits[:, 1] - logits[:, 0]
    positive_margins = margin[positive]
    negative_margins = margin[negative]
    positive_floor = torch.square(torch.relu(
        float(config["positive_logit_margin_floor"]) - positive_margins.min()
    ))
    hard_negative = torch.square(torch.relu(
        negative_margins.max()
        - float(config["hard_negative_logit_margin_ceiling"])
    ))
    positive_suppression = logits[positive, 0] - p2_logits[positive, 0]
    positive_suppression_ceiling = torch.square(torch.relu(
        positive_suppression.max()
        - float(config["positive_suppression_logit_maximum"])
    ))
    scene_separation = torch.square(torch.relu(
        negative_margins.max()
        - positive_margins.min()
        + float(config["scene_separation_logit_margin_minimum"])
    ))
    losses = {
        "asymmetric_cross_entropy": asymmetric,
        "positive_floor": positive_floor,
        "hard_negative": hard_negative,
        "positive_suppression_ceiling": positive_suppression_ceiling,
        "scene_separation": scene_separation,
    }
    total = (
        float(config["proposal_cross_entropy_weight"]) * asymmetric
        + float(config["positive_floor_weight"]) * positive_floor
        + float(config["hard_negative_weight"]) * hard_negative
        + float(config["positive_suppression_weight"]) * positive_suppression_ceiling
        + float(config["scene_separation_weight"]) * scene_separation
    )
    return total, losses


def _p2_result_is_exact(result: dict[str, Any]) -> bool:
    metrics = result.get("selection_metrics", {})
    return bool(
        result.get("status") == "failed_selection"
        and result.get("candidate_id") == "P2"
        and result.get("candidate_consumed") is True
        and result.get("case_level_details_emitted") is False
        and result.get("frozen_p1_weights_preserved") is True
        and result.get("onnx_parity_passed") is True
        and result.get("parent_role_argmax_preserved") is True
        and result.get("passing_threshold_window") == []
        and metrics.get("scene_count") == 128
        and metrics.get("exact_scene_count") == 123
        and metrics.get("true_positives") == 1024
        and metrics.get("false_positives") == 3
        and metrics.get("false_negatives") == 0
        and metrics.get("prohibited_structure_hits") == 3
        and metrics.get("duplicate_region_count") == 0
        and result.get("public_gate_archive_opened") is False
        and result.get("public_gate_evaluations") == 0
        and "cases" not in result
        and "predictions" not in result
    )


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-morphology-veto-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-p2-monotonic-veto-only-v1",
        "objective": "monotonic-veto-hard-negative-max-v1",
        "model_license": "Apache-2.0",
        "seed": 2608182703,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "epochs": 4,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1024,
        "gradient_clip_norm": 2.0,
        "proposal_cross_entropy_weight": 1.0,
        "false_positive_weight": 24.0,
        "positive_logit_margin_floor": 0.2,
        "positive_floor_weight": 8.0,
        "hard_negative_logit_margin_ceiling": -1.0,
        "hard_negative_weight": 32.0,
        "positive_suppression_logit_maximum": 0.1,
        "positive_suppression_weight": 16.0,
        "scene_separation_logit_margin_minimum": 1.0,
        "scene_separation_weight": 4.0,
        "monotonic_acceptance_suppression_only": True,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 10000,
        "detector_prefilter_applied": False,
        "recognition_batch_size": 64,
        "feature_count": FEATURE_COUNT,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "structure_feature_count": STRUCTURE_FEATURE_COUNT,
        "runtime_numeric_precision": "float32",
        "output_logit_scale": OUTPUT_LOGIT_SCALE,
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_result_sha256": P2_RESULT_SHA256,
        "p2_checkpoint_path": P2_CHECKPOINT_PATH.as_posix(),
        "p2_checkpoint_sha256": P2_CHECKPOINT_SHA256,
        "p2_onnx_path": P2_ONNX_PATH.as_posix(),
        "p2_onnx_sha256": P2_ONNX_SHA256,
        "p2_report_path": P2_REPORT_PATH.as_posix(),
        "p2_report_sha256": P2_REPORT_SHA256,
        "selection_thresholds": list(THRESHOLDS),
        "split_seal_path": p2.p1.SEAL_PATH.as_posix(),
        "train_fixture_archive_path": p2.p1.TRAIN_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_path": p2.p1.SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": p2.p1.PUBLIC_ARCHIVE_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "parent_role_argmax_preservation_required": True,
        "selection_evaluation_limit": 1,
        "validation_or_public_pixels_used_for_training": False,
        "case_level_predecessor_evidence_used": False,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V27 P3 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V27 P3 runner source bundle changed")
    if sha256_file(REPO_ROOT / p2.p1.SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V27 split seal changed before P3")
    seal = _read_json(REPO_ROOT / p2.p1.SEAL_PATH)
    if (
        seal.get("schema") != "graphreader.ocr-morphology-veto-split-seal.v1"
        or seal.get("revision") != REVISION
        or seal.get("optimizer_steps_at_freeze") != 0
        or seal.get("selection_evaluations") != 0
        or seal.get("public_evaluations") != 0
        or seal.get("public_execution_authorized") is not False
    ):
        raise RuntimeError("OCR V27 split seal changed before P3")
    head = p2._repository_head()
    if not p2._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V27 split source commit is not an ancestor")
    for relative, expected_hash in seal["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V27 frozen split source changed: {relative}")
    for split, path, key in (
        ("train", p2.p1.TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        ("validation", p2.p1.SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        ("sealed_public", p2.p1.PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    ):
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V27 {split} archive changed before P3")
    for relative, expected_hash in {
        P2_RESULT_PATH: P2_RESULT_SHA256,
        P2_CHECKPOINT_PATH: P2_CHECKPOINT_SHA256,
        P2_ONNX_PATH: P2_ONNX_SHA256,
        P2_REPORT_PATH: P2_REPORT_SHA256,
    }.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V27 P2 evidence changed: {relative}")
    if not _p2_result_is_exact(_read_json(REPO_ROOT / P2_RESULT_PATH)):
        raise RuntimeError("OCR V27 aggregate-only P2 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V27 P3 output already exists")
    ledger = _read_json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(
        (item for item in ledger["revisions"] if item.get("revision") == REVISION), None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_3_preregistered"
        or entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != ["P1", "P2"]
        or entry.get("remaining_unregistered_candidate_ids") != []
    ):
        raise RuntimeError("OCR V27 P3 canonical authorization changed")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V27 P3 output exists: {output_dir}")
    evidence = preflight()
    config = evidence["config"]
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
    phase = "load_frozen_training_fixtures"
    optimizer_steps = 0
    try:
        train_scenes = p2.load_archive(REPO_ROOT / p2.p1.TRAIN_ARCHIVE_PATH)
        p2.p1._validate_stored_split(
            train_scenes, evidence["seal"]["splits"]["train"], "train",
        )
        detector_session = p2._cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = p2._cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        alphabet = p2.read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

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

        phase = "direct_training_feature_crop_and_structure_execution"
        train_values, train_crops, train_labels, train_records, training_evidence = (
            p2.extract_crop_evidence(
                train_scenes,
                detector_runner,
                recognizer_runner,
                alphabet,
                mode="train",
                negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
                recognition_batch_size=int(config["recognition_batch_size"]),
            )
        )
        train_structure = p2.structure_features(train_crops)
        if (
            train_values.shape[1:] != (FEATURE_COUNT,)
            or train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH)
            or train_structure.shape != (len(train_crops), STRUCTURE_FEATURE_COUNT)
        ):
            raise RuntimeError("OCR V27 P3 training tensor contract changed")
        training_evidence["training_structure_tensor_stream_sha256"] = sha256(
            train_structure.tobytes(order="C")
        ).hexdigest()
        train_groups = p2._feature_groups(train_records, len(train_scenes))

        phase = "monotonic_veto_only_training"
        generator = p2._configure(int(config["seed"]))
        model = FrozenP2MonotonicVetoProposalNet(seed=int(config["seed"]))
        model.load_p2_state_dict(_load_p2_state())
        frozen_before = p2._parameter_subset_sha256(model, trainable=False)
        trainable_before = p2._parameter_subset_sha256(model, trainable=True)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if trainable_names != [
            "monotonic_veto.0.bias",
            "monotonic_veto.0.weight",
            "monotonic_veto.2.bias",
            "monotonic_veto.2.weight",
        ]:
            raise RuntimeError("OCR V27 P3 monotonic-veto boundary changed")
        anchor_batches: list[np.ndarray] = []
        with torch.inference_mode():
            model.eval()
            for indices in train_groups:
                anchor_batches.append(model.frozen_p2_output(
                    torch.from_numpy(train_values[indices]).unsqueeze(0),
                    torch.from_numpy(train_crops[indices]).unsqueeze(0),
                    torch.from_numpy(train_structure[indices]).unsqueeze(0),
                ).numpy()[0, :, :2])
        anchor_logits = np.concatenate(anchor_batches)
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        class_weights = p2._balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        loss_checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            sums = {
                "total": 0.0,
                "asymmetric_cross_entropy": 0.0,
                "positive_floor": 0.0,
                "hard_negative": 0.0,
                "positive_suppression_ceiling": 0.0,
                "scene_separation": 0.0,
            }
            for scene_index in torch.randperm(
                len(train_groups), generator=generator,
            ).tolist():
                indices = train_groups[scene_index]
                logits = model(
                    torch.from_numpy(train_values[indices]).unsqueeze(0),
                    torch.from_numpy(train_crops[indices]).unsqueeze(0),
                    torch.from_numpy(train_structure[indices]).unsqueeze(0),
                )[0, :, :2]
                loss, components = _objective(
                    logits,
                    torch.from_numpy(anchor_logits[indices]),
                    torch.from_numpy(train_labels[indices]),
                    class_weights,
                    config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    trainable, max_norm=float(config["gradient_clip_norm"]),
                )
                optimizer.step()
                optimizer_steps += 1
                sums["total"] += float(loss.detach())
                for name, value in components.items():
                    sums[name] += float(value.detach())
            count = len(train_groups)
            loss_checkpoints.append({
                "epoch": epoch + 1,
                **{name: value / count for name, value in sums.items()},
            })
            print(
                f"OCR V27 P3 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V27 P3 optimizer-step count changed")
        frozen_after = p2._parameter_subset_sha256(model, trainable=False)
        trainable_after = p2._parameter_subset_sha256(model, trainable=True)
        if frozen_after != frozen_before or trainable_after == trainable_before:
            raise RuntimeError("OCR V27 P3 parameter-change boundary failed")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-morphology-veto-proposal-v27-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-morphology-veto-proposal-v27-p3.onnx"
        first = train_groups[0]
        model.eval()
        p2.p1._export(
            model,
            torch.from_numpy(train_values[first]).unsqueeze(0),
            torch.from_numpy(train_crops[first]).unsqueeze(0),
            torch.from_numpy(train_structure[first]).unsqueeze(0),
            onnx_path,
        )
        candidate_session = p2.p1._candidate_session(onnx_path)

        phase = "single_visible_selection"
        selection_scenes = p2.load_archive(REPO_ROOT / p2.p1.SELECTION_ARCHIVE_PATH)
        p2.p1._validate_stored_split(
            selection_scenes, evidence["seal"]["splits"]["validation"], "validation",
        )
        selection_values, selection_crops, _, selection_records, selection_evidence = (
            p2.extract_crop_evidence(
                selection_scenes,
                detector_runner,
                recognizer_runner,
                alphabet,
                mode="train",
                negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
                recognition_batch_size=int(config["recognition_batch_size"]),
            )
        )
        selection_structure = p2.structure_features(selection_crops)
        selection_groups = p2._feature_groups(selection_records, len(selection_scenes))
        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        structure_inputs = sha256()
        candidate_outputs = sha256()
        p2_outputs = sha256()
        parity_error = 0.0
        role_argmax_mismatches = 0
        monotonic_violations = 0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                structure = np.ascontiguousarray(selection_structure[indices][None, ...])
                value_tensor = torch.from_numpy(values)
                crop_tensor = torch.from_numpy(crops)
                structure_tensor = torch.from_numpy(structure)
                expected_output = model(value_tensor, crop_tensor, structure_tensor).numpy()
                p2_output = model.frozen_p2_output(
                    value_tensor, crop_tensor, structure_tensor,
                ).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                    "structure_features": structure,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                role_argmax_mismatches += int(np.count_nonzero(
                    np.argmax(actual_output[:, :, 2:], axis=2)
                    != np.argmax(p2_output[:, :, 2:], axis=2)
                ))
                monotonic_violations += int(np.count_nonzero(
                    (actual_output[:, :, 0] < p2_output[:, :, 0])
                    | (actual_output[:, :, 1] > p2_output[:, :, 1])
                ))
                outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                structure_inputs.update(structure.tobytes(order="C"))
                candidate_outputs.update(actual_output.tobytes(order="C"))
                p2_outputs.update(p2_output.tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = p2._calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_structure_input_tensor_stream_sha256": structure_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "p2_output_tensor_stream_sha256": p2_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "parent_role_argmax_mismatch_count": role_argmax_mismatches,
            "monotonic_acceptance_suppression_violation_count": monotonic_violations,
        })
        comparisons = p2.evaluate_thresholds(
            selection_scenes,
            calibrated_records,
            flat_output[:, :2],
            tuple(float(value) for value in config["selection_thresholds"]),
            selection_evidence,
        )
        robust = p2.select_robust_window(comparisons)
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
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        roles_preserved = role_argmax_mismatches == 0
        monotonic_passed = monotonic_violations == 0
        frozen_preserved = frozen_before == frozen_after
        passed = (
            robust is not None
            and parity_passed
            and roles_preserved
            and monotonic_passed
            and frozen_preserved
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-morphology-veto-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": True,
            "objective": config["objective"],
            "training_evidence": training_evidence,
            "loss_checkpoints": loss_checkpoints,
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_stream_sha256_before": frozen_before,
            "frozen_parameter_stream_sha256_after": frozen_after,
            "frozen_p2_weights_preserved": frozen_preserved,
            "p2_checkpoint_sha256": P2_CHECKPOINT_SHA256,
            "parent_role_argmax_mismatch_count": role_argmax_mismatches,
            "parent_role_argmax_preserved": roles_preserved,
            "monotonic_acceptance_suppression_violation_count": monotonic_violations,
            "monotonic_acceptance_suppression_passed": monotonic_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "selection_evidence": selection_evidence,
            "training_authorization": authorization.binding,
            "split_seal_sha256": config["split_seal_sha256"],
            "train_fixture_archive_sha256": config["train_fixture_archive_sha256"],
            "selection_fixture_archive_sha256": config["selection_fixture_archive_sha256"],
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "private_validation_authorized": False,
            "manifest_creation_authorized": False,
            "model_store_promotion_authorized": False,
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization, status=str(report["status"]), report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-morphology-veto-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "training_authorization": authorization.binding,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "production_approval": False,
            "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(
            authorization, status="failed_runner", report_sha256=sha256_file(report_path),
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight()
        print(json.dumps({
            "head": evidence["head"],
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
            "ready": True,
        }, sort_keys=True))
        return 0
    report = train_candidate(REPO_ROOT / CANONICAL_OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "onnx_parity_maximum_absolute_error": report[
            "onnx_parity_maximum_absolute_error"
        ],
        "parent_role_argmax_mismatch_count": report[
            "parent_role_argmax_mismatch_count"
        ],
        "monotonic_acceptance_suppression_violation_count": report[
            "monotonic_acceptance_suppression_violation_count"
        ],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT", "CONFIG_PATH", "RUNNER_SOURCE_PATHS",
    "_objective", "preflight", "train_candidate",
]
