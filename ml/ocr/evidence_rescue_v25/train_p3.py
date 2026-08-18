# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use multiscale spatial residual training for OCR V25 P3."""

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
from ml.markers.training_budget import (
    CANONICAL_LEDGER_PATH,
    acquire_training_candidate,
    complete_training_candidate,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _balanced_class_weights,
    _calibrated_records,
    _configure,
    _cpu_session,
    _export,
    _feature_groups,
    _is_ancestor,
    _read_json,
    _repository_head,
)

from .dataset import load_archive
from .model_p3 import FrozenParentMultiscaleSpatialResidualNet
from .pipeline import extract_crop_evidence
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    PARENT_CHECKPOINT_PATH,
    PARENT_CHECKPOINT_SHA256,
    PARENT_ONNX_PATH,
    PARENT_ONNX_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    SEED,
    TASK,
    THRESHOLDS,
)
from .train_p1 import _load_parent_state, _validate_stored_split


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/evidence_rescue_v25")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v25-public.zip")
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "model_p3.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    ROOT / "train_p3.py",
    P1_RESULT_PATH,
    P2_RESULT_PATH,
    Path("ml/ocr/crop_evidence_role_anchor_v24/model_p2.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/pipeline.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/protocol.py"),
    Path("ml/ocr/crop_evidence_role_anchor_v24/train_p1.py"),
    Path("ml/ocr/role_anchor_set_v23/model.py"),
    Path("ml/ocr/role_anchor_set_v23/protocol.py"),
    Path("ml/ocr/margin_calibrator_v20/dataset.py"),
    Path("ml/ocr/margin_calibrator_v20/pipeline.py"),
    Path("ml/ocr/margin_calibrator_v20/protocol.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/dataset.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/protocol.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _parameter_stream_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(parameter.detach().numpy()).tobytes(order="C"))
    return digest.hexdigest()


def _proposal_residual_objective(
    candidate_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base = nn.functional.cross_entropy(candidate_logits, targets, weight=class_weights)
    candidate_margin = candidate_logits[:, 1] - candidate_logits[:, 0]
    teacher_margin = teacher_logits[:, 1] - teacher_logits[:, 0]
    positive = targets == 1
    negative = targets == 0
    if not torch.any(positive) or not torch.any(negative):
        raise RuntimeError("OCR V25 P3 requires positive and negative proposals per scene")
    positive_floor = torch.relu(
        float(config["positive_logit_margin_floor"]) - candidate_margin[positive]
    ).max()
    negative_ceiling = torch.relu(
        candidate_margin[negative] - float(config["negative_logit_margin_ceiling"])
    ).max()
    teacher_positive_drop = torch.relu(
        teacher_margin[positive] - candidate_margin[positive]
    ).max()
    teacher_negative_worsening = torch.relu(
        candidate_margin[negative] - teacher_margin[negative]
    ).max()
    separation = torch.relu(
        candidate_margin[negative].max()
        - candidate_margin[positive].min()
        + float(config["scene_separation_logit_margin_minimum"])
    )
    losses = {
        "cross_entropy": base,
        "positive_floor": positive_floor,
        "negative_ceiling": negative_ceiling,
        "teacher_positive_drop": teacher_positive_drop,
        "teacher_negative_worsening": teacher_negative_worsening,
        "scene_separation": separation,
    }
    total = (
        float(config["proposal_cross_entropy_weight"]) * base
        + float(config["positive_floor_weight"]) * positive_floor
        + float(config["negative_ceiling_weight"]) * negative_ceiling
        + float(config["teacher_positive_weight"]) * teacher_positive_drop
        + float(config["teacher_negative_weight"]) * teacher_negative_worsening
        + float(config["scene_separation_weight"]) * separation
    )
    return total, losses


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-evidence-rescue-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-v25-p1-plus-multiscale-spatial-residual-v1",
        "objective": "teacher-preserving-multiscale-spatial-residual-v1",
        "model_license": "Apache-2.0",
        "seed": SEED + 2,
        "learning_rate": 0.0005,
        "weight_decay": 0.001,
        "epochs": 5,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1280,
        "spatial_encoder_channels": [8, 16],
        "spatial_encoder_kernel": 3,
        "spatial_encoder_stride": 2,
        "spatial_pool_height": 2,
        "spatial_pool_width": 2,
        "residual_hidden_widths": [96, 48],
        "dropout_probability": 0.1,
        "gradient_clip_norm": 5.0,
        "residual_scale": 1.0,
        "proposal_cross_entropy_weight": 1.0,
        "positive_logit_margin_floor": 2.2,
        "positive_floor_weight": 3.0,
        "negative_logit_margin_ceiling": -2.2,
        "negative_ceiling_weight": 3.0,
        "teacher_positive_weight": 1.0,
        "teacher_negative_weight": 2.0,
        "scene_separation_logit_margin_minimum": 4.4,
        "scene_separation_weight": 1.0,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 10000,
        "detector_prefilter_applied": False,
        "recognition_batch_size": 64,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "parent_checkpoint_path": PARENT_CHECKPOINT_PATH,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_onnx_path": PARENT_ONNX_PATH,
        "parent_onnx_sha256": PARENT_ONNX_SHA256,
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "p1_result_sha256": "700e1b659a78bc48ebfd893dc66d5b23f082c340857c64da4f7fd752c5f7dcf1",
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_result_sha256": "266df7f7e70f767a455b92acb06adc43bec7763e9404256b84dc4e896cea45c1",
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "split_seal_path": SEAL_PATH.as_posix(),
        "split_source_commit": "9805c2db397e2b7857093b3292cf115ddb6d559b",
        "split_source_bundle_sha256": "624bfcaf73fbf791176fd5aa23ceb9e57f35296163ddd9453abb40f0d7c3852d",
        "train_fixture_archive_path": TRAIN_ARCHIVE_PATH.as_posix(),
        "train_fixture_archive_sha256": "63f9d63a34dca66fb65047355ba2a9e2381ea918c476473411e20f48f243bcc1",
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_sha256": "20284a9a259409cf9b94550bfed244fd59933a287f1fe2a13d2000ff8c354f9d",
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_sha256": "bab5eaa67d7d7427334523a2db3780b516b1e822bb017c0a154e2f029d46ec40",
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "selection_evaluation_limit": 1,
        "validation_or_public_pixels_used_for_training": False,
        "case_level_selection_evidence_used_for_design": False,
        "p2_case_level_evidence_used_for_design": False,
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
            raise RuntimeError(f"OCR V25 P3 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V25 P3 thresholds changed")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V25 P3 runner source bundle changed")
    exact_inputs = {
        PROTOCOL_PATH: config["protocol_sha256"],
        SEAL_PATH: config["split_seal_sha256"],
        P1_RESULT_PATH: config["p1_result_sha256"],
        P2_RESULT_PATH: config["p2_result_sha256"],
        Path(DETECTOR_PATH): DETECTOR_SHA256,
        Path(RECOGNIZER_PATH): RECOGNIZER_SHA256,
        Path(RECOGNIZER_YAML_PATH): RECOGNIZER_YAML_SHA256,
        Path(PARENT_CHECKPOINT_PATH): PARENT_CHECKPOINT_SHA256,
        Path(PARENT_ONNX_PATH): PARENT_ONNX_SHA256,
        TRAIN_ARCHIVE_PATH: config["train_fixture_archive_sha256"],
        SELECTION_ARCHIVE_PATH: config["selection_fixture_archive_sha256"],
        PUBLIC_ARCHIVE_PATH: config["public_fixture_archive_sha256"],
    }
    for relative, expected_hash in exact_inputs.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V25 P3 frozen input changed: {relative.as_posix()}")
    p2 = _read_json(REPO_ROOT / P2_RESULT_PATH)
    metrics = p2.get("selection_metrics", {})
    if (
        p2.get("status") != "failed_selection"
        or p2.get("candidate_consumed") is not True
        or p2.get("case_level_details_emitted") is not False
        or metrics.get("exact_scene_count") != 112
        or metrics.get("false_positives") != 1
        or metrics.get("false_negatives") != 8
        or metrics.get("prohibited_structure_hits") != 1
        or metrics.get("duplicate_region_count") != 0
        or p2.get("parent_roles_preserved") is not True
        or p2.get("public_gate_archive_opened") is not False
        or p2.get("public_gate_evaluations") != 0
        or "cases" in p2
        or "predictions" in p2
    ):
        raise RuntimeError("OCR V25 P3 aggregate-only P2 trigger changed")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V25 split source commit is not an ancestor")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V25 P3 output already exists")
    ledger = _read_json(REPO_ROOT / CANONICAL_LEDGER_PATH)
    entry = next(
        (
            item for item in ledger.get("revisions", [])
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_3_preregistered"
        or entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != ["P1", "P2"]
    ):
        raise RuntimeError("OCR V25 P3 canonical single-candidate authorization changed")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V25 P3 output exists: {output_dir}")
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
        _validate_stored_split(train_scenes, evidence["seal"]["splits"]["train"], "train")
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
            raise RuntimeError("OCR V25 P3 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V25 P3 training crop shape changed")
        train_groups = _feature_groups(train_records, len(train_scenes))
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V25 P3 incomplete training stream: {key}")

        phase = "pooled_evidence_residual_training"
        generator = _configure(int(config["seed"]))
        model = FrozenParentMultiscaleSpatialResidualNet(
            seed=int(config["seed"]),
            residual_scale=float(config["residual_scale"]),
            dropout_probability=float(config["dropout_probability"]),
        )
        model.load_parent_state_dict(_load_parent_state())
        frozen_before = _parameter_stream_sha256(model.parent)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if not trainable or any(name.startswith("parent.") for name in trainable_names):
            raise RuntimeError("OCR V25 P3 frozen-parent boundary changed")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        class_weights = _balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        loss_checkpoints: list[dict[str, float | int]] = []
        model.train()
        model.parent.eval()
        for epoch in range(int(config["epochs"])):
            sums = {
                "total": 0.0,
                "cross_entropy": 0.0,
                "positive_floor": 0.0,
                "negative_ceiling": 0.0,
                "teacher_positive_drop": 0.0,
                "teacher_negative_worsening": 0.0,
                "scene_separation": 0.0,
            }
            for scene_index in torch.randperm(
                len(train_groups), generator=generator,
            ).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[indices]).unsqueeze(0)
                targets = torch.from_numpy(train_labels[indices])
                with torch.no_grad():
                    teacher = model.parent(values, crops)[0, :, :2]
                candidate = model(values, crops)[0, :, :2]
                loss, components = _proposal_residual_objective(
                    candidate, teacher, targets, class_weights, config,
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
                f"OCR V25 P3 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V25 P3 optimizer-step count changed")
        frozen_after = _parameter_stream_sha256(model.parent)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V25 P3 modified the frozen V24 parent")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-evidence-rescue-v25-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-evidence-rescue-v25-p3.onnx"
        first = train_groups[0]
        example_values = torch.from_numpy(train_values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(train_crops[first]).unsqueeze(0)
        model.eval()
        _export(model, example_values, example_crops, onnx_path)
        candidate_session = _cpu_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops",
        }:
            raise RuntimeError("OCR V25 P3 ONNX input identity changed")

        phase = "single_visible_selection"
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        _validate_stored_split(
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
        selection_groups = _feature_groups(selection_records, len(selection_scenes))
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V25 P3 incomplete selection stream: {key}")

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
                parent_output = model.parent(torch_values, torch_crops).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                role_error = max(
                    role_error,
                    float(np.max(np.abs(
                        expected_output[:, :, 2:] - parent_output[:, :, 2:],
                    ))),
                )
                outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual_output).tobytes(order="C"))
                parent_outputs.update(np.ascontiguousarray(parent_output).tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "parent_output_tensor_stream_sha256": parent_outputs.hexdigest(),
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
            "schema": "graphreader.ocr-evidence-rescue-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": True,
            "objective": config["objective"],
            "objective_configuration": {
                key: config[key] for key in (
                    "proposal_cross_entropy_weight",
                    "positive_logit_margin_floor",
                    "positive_floor_weight",
                    "negative_logit_margin_ceiling",
                    "negative_ceiling_weight",
                    "teacher_positive_weight",
                    "teacher_negative_weight",
                    "scene_separation_logit_margin_minimum",
                    "scene_separation_weight",
                    "residual_scale",
                )
            },
            "architecture_configuration": {
                key: config[key] for key in (
                    "spatial_encoder_channels",
                    "spatial_encoder_kernel",
                    "spatial_encoder_stride",
                    "spatial_pool_height",
                    "spatial_pool_width",
                    "residual_hidden_widths",
                    "dropout_probability",
                    "gradient_clip_norm",
                )
            },
            "training_evidence": training_evidence,
            "loss_checkpoints": loss_checkpoints,
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_stream_sha256_before": frozen_before,
            "frozen_parameter_stream_sha256_after": frozen_after,
            "frozen_parent_preserved": frozen_preserved,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
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
            "p1_result_sha256": config["p1_result_sha256"],
            "p2_result_sha256": config["p2_result_sha256"],
            "split_seal_sha256": config["split_seal_sha256"],
            "train_fixture_archive_sha256": config["train_fixture_archive_sha256"],
            "selection_fixture_archive_sha256": config[
                "selection_fixture_archive_sha256"
            ],
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
            "schema": "graphreader.ocr-evidence-rescue-failure.v1",
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
        "parent_role_maximum_absolute_error": report[
            "parent_role_maximum_absolute_error"
        ],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT",
    "CONFIG_PATH",
    "RUNNER_SOURCE_PATHS",
    "_proposal_residual_objective",
    "preflight",
    "train_candidate",
]
