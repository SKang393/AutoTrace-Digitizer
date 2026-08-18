# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use bounded-margin hard-negative repair for OCR V26 P2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from . import train_p1 as p1
from .dataset import load_archive
from .model_p2 import FrozenP1BoundedMarginProposalNet
from .pipeline import extract_crop_evidence
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    ROLE_PARENT_CHECKPOINT_PATH,
    ROLE_PARENT_CHECKPOINT_SHA256,
    ROLE_PARENT_ONNX_PATH,
    ROLE_PARENT_ONNX_SHA256,
    TASK,
    THRESHOLDS,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/scene_topology_proposal_v26")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v26-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v26-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v26-public.zip")
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P1_RESULT_SHA256 = "1c1a3041320a4fbe6fcb917355f4b2c60b10b43ba4ce9bd7d1124d115c7e5338"
P1_CHECKPOINT_PATH = ROOT / "artifacts/P1-run/graph-text-scene-topology-proposal-v26-p1.pt"
P1_CHECKPOINT_SHA256 = "29fe9349efe90216900ff6cc29de6374ae63a8f4b609a54d011ab5a2d6f66d89"
P1_ONNX_PATH = ROOT / "artifacts/P1-run/graph-text-scene-topology-proposal-v26-p1.onnx"
P1_ONNX_SHA256 = "d6cb6910faac97997f30f8e7f842aa63dca818bb102b601c4ba9545bdf88c241"
P1_REPORT_PATH = ROOT / "artifacts/P1-run/candidate-report.json"
P1_REPORT_SHA256 = "d962cd133b943036d84d6221e8d8d647c8182b8bed6590a61afd1bc77d4ec32b"
RUNNER_SOURCE_PATHS = (
    *p1.RUNNER_SOURCE_PATHS,
    P1_RESULT_PATH,
    ROOT / "model_p2.py",
    ROOT / "train_p2.py",
)


def _frozen_parameter_stream_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(parameter.detach().numpy()).tobytes(order="C"))
    return digest.hexdigest()


def _load_p1_state() -> dict[str, torch.Tensor]:
    payload = torch.load(
        REPO_ROOT / P1_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("OCR V26 P1 checkpoint state is missing")
    return state


def _proposal_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base = nn.functional.cross_entropy(logits, targets, weight=class_weights)
    margin = logits[:, 1] - logits[:, 0]
    positive = targets == 1
    negative = targets == 0
    if not torch.any(positive) or not torch.any(negative):
        raise RuntimeError("OCR V26 P2 requires positive and negative proposals per scene")
    positive_margins = margin[positive]
    negative_margins = margin[negative]
    positive_floor = torch.relu(
        float(config["positive_logit_margin_floor"]) - positive_margins
    ).max()
    negative_ceiling = torch.relu(
        negative_margins - float(config["negative_logit_margin_ceiling"])
    ).max()
    hard_count = max(
        1,
        int(math.ceil(float(config["hard_negative_fraction"]) * negative_margins.numel())),
    )
    hard_negative_margins = torch.topk(
        negative_margins,
        k=min(hard_count, negative_margins.numel()),
    ).values
    hard_negative = torch.square(torch.relu(
        hard_negative_margins
        - float(config["hard_negative_logit_margin_ceiling"])
    )).mean()
    margin_target = 0.5 * (
        torch.square(
            positive_margins - float(config["positive_logit_margin_target"])
        ).mean()
        + torch.square(
            negative_margins - float(config["negative_logit_margin_target"])
        ).mean()
    )
    losses = {
        "cross_entropy": base,
        "positive_floor": positive_floor,
        "negative_ceiling": negative_ceiling,
        "hard_negative": hard_negative,
        "margin_target": margin_target,
    }
    total = (
        float(config["proposal_cross_entropy_weight"]) * base
        + float(config["positive_floor_weight"]) * positive_floor
        + float(config["negative_ceiling_weight"]) * negative_ceiling
        + float(config["hard_negative_weight"]) * hard_negative
        + float(config["margin_target_weight"]) * margin_target
    )
    return total, losses


def preflight() -> dict[str, Any]:
    config = p1._read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-scene-topology-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-p1-features-bounded-margin-proposal-v1",
        "objective": "bounded-margin-training-hard-negative-recalibration-v1",
        "model_license": "Apache-2.0",
        "seed": 2608182602,
        "learning_rate": 0.00015,
        "weight_decay": 0.001,
        "epochs": 3,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1152,
        "dropout_probability": 0.08,
        "gradient_clip_norm": 5.0,
        "proposal_cross_entropy_weight": 1.0,
        "positive_logit_margin_floor": 1.15,
        "negative_logit_margin_ceiling": -0.7,
        "positive_floor_weight": 6.0,
        "negative_ceiling_weight": 6.0,
        "hard_negative_fraction": 0.1,
        "hard_negative_logit_margin_ceiling": -1.0,
        "hard_negative_weight": 8.0,
        "positive_logit_margin_target": 1.5,
        "negative_logit_margin_target": -1.5,
        "margin_target_weight": 0.25,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 10000,
        "detector_prefilter_applied": False,
        "recognition_batch_size": 64,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "trainable_scope": "proposal_head_only",
        "frozen_scope": "p1_role_parent_crop_stem_crop_projection_evidence_projection",
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "role_parent_checkpoint_path": ROLE_PARENT_CHECKPOINT_PATH,
        "role_parent_checkpoint_sha256": ROLE_PARENT_CHECKPOINT_SHA256,
        "role_parent_onnx_path": ROLE_PARENT_ONNX_PATH,
        "role_parent_onnx_sha256": ROLE_PARENT_ONNX_SHA256,
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "p1_result_sha256": P1_RESULT_SHA256,
        "p1_checkpoint_path": P1_CHECKPOINT_PATH.as_posix(),
        "p1_checkpoint_sha256": P1_CHECKPOINT_SHA256,
        "p1_onnx_path": P1_ONNX_PATH.as_posix(),
        "p1_onnx_sha256": P1_ONNX_SHA256,
        "p1_report_path": P1_REPORT_PATH.as_posix(),
        "p1_report_sha256": P1_REPORT_SHA256,
        "p1_aggregate_design_basis": "P1 aggregate exact-scene, true-positive, false-positive, false-negative, prohibited-hit, duplicate, recognition, role, threshold-window, and parity metrics only",
        "p1_case_detail_or_pixels_used": False,
        "selection_thresholds": list(THRESHOLDS),
        "split_seal_path": SEAL_PATH.as_posix(),
        "train_fixture_archive_path": TRAIN_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "selection_evaluation_limit": 1,
        "selection_evaluations_before_candidate": 1,
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
            raise RuntimeError(f"OCR V26 P2 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V26 P2 runner source bundle changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V26 split seal changed before P2")
    seal = p1._read_json(REPO_ROOT / SEAL_PATH)
    if (
        seal.get("schema") != "graphreader.ocr-scene-topology-split-seal.v1"
        or seal.get("revision") != REVISION
        or seal.get("public_evaluations") != 0
        or seal.get("public_execution_authorized") is not False
        or seal.get("private_data") is not False
        or seal.get("chandler_used") is not False
    ):
        raise RuntimeError("OCR V26 split seal state changed before P2")
    head = p1._repository_head()
    if not p1._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V26 split source commit is not an ancestor")
    if seal.get("source_bundle_sha256") != config.get("split_source_bundle_sha256"):
        raise RuntimeError("OCR V26 split source bundle changed")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V26 split source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V26 frozen split source changed: {relative}")
    archive_bindings = {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (path, key) in archive_bindings.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V26 {split} archive changed before P2")
    exact_inputs = {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        ROLE_PARENT_CHECKPOINT_PATH: ROLE_PARENT_CHECKPOINT_SHA256,
        ROLE_PARENT_ONNX_PATH: ROLE_PARENT_ONNX_SHA256,
        P1_RESULT_PATH.as_posix(): P1_RESULT_SHA256,
        P1_CHECKPOINT_PATH.as_posix(): P1_CHECKPOINT_SHA256,
        P1_ONNX_PATH.as_posix(): P1_ONNX_SHA256,
        P1_REPORT_PATH.as_posix(): P1_REPORT_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V26 P2 frozen input changed: {relative}")
    p1_result = p1._read_json(REPO_ROOT / P1_RESULT_PATH)
    metrics = p1_result.get("selection_metrics", {})
    if (
        p1_result.get("candidate_id") != "P1"
        or p1_result.get("candidate_consumed") is not True
        or p1_result.get("status") != "failed_selection"
        or p1_result.get("case_level_details_emitted") is not False
        or p1_result.get("report_sha256") != P1_REPORT_SHA256
        or p1_result.get("checkpoint_sha256") != P1_CHECKPOINT_SHA256
        or p1_result.get("onnx_sha256") != P1_ONNX_SHA256
        or metrics.get("scene_count") != 128
        or metrics.get("exact_scene_count") != 118
        or metrics.get("true_positives") != 1024
        or metrics.get("false_positives") != 5
        or metrics.get("false_negatives") != 0
        or metrics.get("duplicate_region_count") != 0
        or metrics.get("prohibited_structure_hits") != 5
        or p1_result.get("onnx_parity_maximum_absolute_error") != 1.1444091796875e-05
        or p1_result.get("passing_threshold_window") != []
        or p1_result.get("public_gate_archive_opened") is not False
        or p1_result.get("public_gate_evaluations") != 0
        or "cases" in p1_result
        or "predictions" in p1_result
    ):
        raise RuntimeError("OCR V26 P2 aggregate-only P1 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V26 P2 output already exists")
    ledger = p1._read_json(
        REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    )
    entry = next(
        (
            item for item in ledger.get("revisions", [])
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_2_preregistered"
        or entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != ["P1"]
        or entry.get("remaining_unregistered_candidate_ids") != ["P3"]
        or entry.get("selection_evaluations") != 1
        or entry.get("public_gate_authorized") is not False
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
    ):
        raise RuntimeError("OCR V26 P2 canonical authorization changed")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V26 P2 output exists: {output_dir}")
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
        train_scenes = load_archive(REPO_ROOT / TRAIN_ARCHIVE_PATH)
        p1._validate_stored_split(train_scenes, evidence["seal"]["splits"]["train"], "train")
        detector_session = p1._cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = p1._cpu_session(REPO_ROOT / RECOGNIZER_PATH)
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

        phase = "direct_training_feature_and_crop_execution"
        train_values, train_crops, train_labels, train_records, training_evidence = (
            extract_crop_evidence(
                train_scenes,
                detector_runner,
                recognizer_runner,
                alphabet,
                mode="train",
                negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
                recognition_batch_size=int(config["recognition_batch_size"]),
            )
        )
        if train_values.shape[1:] != (FEATURE_COUNT,):
            raise RuntimeError("OCR V26 P2 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V26 P2 training crop shape changed")
        train_groups = p1._feature_groups(train_records, len(train_scenes))
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V26 P2 incomplete training stream: {key}")

        phase = "bounded_margin_hard_negative_training"
        generator = p1._configure(int(config["seed"]))
        model = FrozenP1BoundedMarginProposalNet(
            seed=int(config["seed"]),
            dropout_probability=float(config["dropout_probability"]),
        )
        model.load_p1_state_dict(_load_p1_state())
        frozen_before = _frozen_parameter_stream_sha256(model)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if not trainable or any(
            not name.startswith("proposal_head.") for name in trainable_names
        ):
            raise RuntimeError("OCR V26 P2 trainable scope changed")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        class_weights = p1._balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        loss_checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            sums = {
                "total": 0.0,
                "cross_entropy": 0.0,
                "positive_floor": 0.0,
                "negative_ceiling": 0.0,
                "hard_negative": 0.0,
                "margin_target": 0.0,
            }
            for scene_index in torch.randperm(
                len(train_groups), generator=generator,
            ).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[indices]).unsqueeze(0)
                targets = torch.from_numpy(train_labels[indices])
                candidate = model(values, crops)[0, :, :2]
                loss, components = _proposal_objective(
                    candidate, targets, class_weights, config,
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
                f"OCR V26 P2 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V26 P2 optimizer-step count changed")
        frozen_after = _frozen_parameter_stream_sha256(model)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V26 P2 modified the frozen P1 feature or role weights")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-scene-topology-proposal-v26-p2.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-scene-topology-proposal-v26-p2.onnx"
        first = train_groups[0]
        example_values = torch.from_numpy(train_values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(train_crops[first]).unsqueeze(0)
        model.eval()
        p1._export(model, example_values, example_crops, onnx_path)
        candidate_session = p1._cpu_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops",
        }:
            raise RuntimeError("OCR V26 P2 ONNX input identity changed")

        phase = "single_visible_selection"
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        p1._validate_stored_split(
            selection_scenes, evidence["seal"]["splits"]["validation"], "validation",
        )
        selection_values, selection_crops, _, selection_records, selection_evidence = (
            extract_crop_evidence(
                selection_scenes,
                detector_runner,
                recognizer_runner,
                alphabet,
                mode="train",
                negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
                recognition_batch_size=int(config["recognition_batch_size"]),
            )
        )
        selection_groups = p1._feature_groups(selection_records, len(selection_scenes))
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V26 P2 incomplete selection stream: {key}")

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        candidate_outputs = sha256()
        parent_outputs = sha256()
        parity_error = 0.0
        role_error = 0.0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                torch_values = torch.from_numpy(values)
                torch_crops = torch.from_numpy(crops)
                expected_output = model(torch_values, torch_crops).numpy()
                parent_output = model.role_parent(torch_values, torch_crops).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                role_error = max(
                    role_error,
                    float(np.max(np.abs(expected_output[:, :, 2:] - parent_output[:, :, 2:]))),
                )
                outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual_output).tobytes(order="C"))
                parent_outputs.update(np.ascontiguousarray(parent_output).tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = p1._calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "role_parent_output_tensor_stream_sha256": parent_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "parent_role_maximum_absolute_error": role_error,
        })
        comparisons = evaluate_thresholds(
            selection_scenes,
            calibrated_records,
            flat_output[:, :2],
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
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        roles_preserved = role_error == 0.0
        frozen_preserved = frozen_before == frozen_after
        passed = robust is not None and parity_passed and roles_preserved and frozen_preserved
        report: dict[str, object] = {
            "schema": "graphreader.ocr-scene-topology-candidate-report.v1",
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
            "frozen_p1_features_and_role_parent_preserved": frozen_preserved,
            "p1_result_sha256": P1_RESULT_SHA256,
            "p1_checkpoint_sha256": P1_CHECKPOINT_SHA256,
            "p1_onnx_sha256": P1_ONNX_SHA256,
            "p1_report_sha256": P1_REPORT_SHA256,
            "parent_role_maximum_absolute_error": role_error,
            "parent_roles_preserved": roles_preserved,
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
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-scene-topology-failure.v1",
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
            authorization,
            status="failed_runner",
            report_sha256=sha256_file(report_path),
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
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
        "parent_role_maximum_absolute_error": report["parent_role_maximum_absolute_error"],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT",
    "CONFIG_PATH",
    "RUNNER_SOURCE_PATHS",
    "_proposal_objective",
    "preflight",
    "train_candidate",
]
