# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use frozen-P1 relational role repair for OCR V28 P2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _balanced_class_weights,
    _calibrated_records,
    _configure,
    _cpu_session,
    _is_ancestor,
    _read_json,
    _repository_head,
    _role_targets,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive
from .model_p2 import FrozenP1RelationalRoleResidualNet
from .pipeline import extract_relational_evidence
from .prepare_split import SOURCE_PATHS
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
    RELATION_FEATURE_COUNT,
    REVISION,
    ROLE_ORDER,
    SEED,
    TASK,
    THRESHOLDS,
)
from .train_p1 import (
    _candidate_session,
    _export,
    _parameter_stream_sha256,
    _validate_stored_split,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/relational_neighborhood_proposal_v28")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
PREREGISTRATION_PATH = ROOT / "P2_PREREGISTRATION.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P1_RESULT_SHA256 = "aa680630cdc1d94941d6864ec0ad8de5c0a9ad7763d37b18b193ad43339dc0be"
P1_CHECKPOINT_PATH = (
    ROOT / "artifacts/P1-run/graph-text-relational-neighborhood-proposal-v28-p1.pt"
)
P1_CHECKPOINT_SHA256 = "7c42fff45fcd116e0f020cab4cc8e9a238d70dd943c302a5e3884cd1b570930c"
P1_ONNX_PATH = (
    ROOT / "artifacts/P1-run/graph-text-relational-neighborhood-proposal-v28-p1.onnx"
)
P1_ONNX_SHA256 = "788fe3ff7737b3a32db26533fa343477ef4f2d1db73a83f634eba6fbf6054867"
P1_REPORT_PATH = ROOT / "artifacts/P1-run/candidate-report.json"
P1_REPORT_SHA256 = "af7d6ca29d374880f479977c6b6f740193f549593b0a9a993e5e06c8b0b5c618"
P1_SELECTION_OUTPUT_STREAM_SHA256 = (
    "e1d51933f8711c0dc3b5408c6c24363a46375c66f07dedfbaea87877a5a0649e"
)
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v28-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v28-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v28-public.zip")
RUNNER_SOURCE_PATHS = tuple(dict.fromkeys((
    *SOURCE_PATHS,
    P1_RESULT_PATH,
    PREREGISTRATION_PATH,
    ROOT / "model_p2.py",
    ROOT / "train_p2.py",
)))
TRAINABLE_PARAMETER_NAMES = (
    "role_residual.0.bias",
    "role_residual.0.weight",
    "role_residual.2.bias",
    "role_residual.2.weight",
    "role_residual.4.bias",
    "role_residual.4.weight",
)


def _module_parameter_stream_sha256(model: nn.Module) -> str:
    digest = sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(
            parameter.detach().numpy(),
        ).tobytes(order="C"))
    return digest.hexdigest()


def _role_margin(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    target = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    mask = nn.functional.one_hot(targets, num_classes=logits.shape[1]).bool()
    other = logits.masked_fill(mask, torch.finfo(logits.dtype).min).max(dim=1).values
    return target - other


def _role_residual_objective(
    candidate_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if candidate_logits.shape != teacher_logits.shape:
        raise RuntimeError("OCR V28 P2 candidate and teacher role shapes differ")
    if candidate_logits.ndim != 2 or candidate_logits.shape[1] != len(ROLE_ORDER):
        raise RuntimeError("OCR V28 P2 role-logit shape changed")
    per_example = nn.functional.cross_entropy(
        candidate_logits, targets, weight=class_weights, reduction="none",
    )
    teacher_correct = teacher_logits.argmax(dim=1) == targets
    teacher_wrong = ~teacher_correct
    base = per_example.mean()
    error_focus = (
        per_example[teacher_wrong].mean()
        if torch.any(teacher_wrong)
        else per_example.sum() * 0.0
    )
    candidate_margin = _role_margin(candidate_logits, targets)
    teacher_margin = _role_margin(teacher_logits, targets)
    preservation = (
        torch.relu(
            teacher_margin[teacher_correct]
            - float(config["teacher_correct_margin_drop_maximum"])
            - candidate_margin[teacher_correct]
        ).max()
        if torch.any(teacher_correct)
        else candidate_margin.sum() * 0.0
    )
    margin_floor = torch.relu(
        float(config["true_role_logit_margin_floor"]) - candidate_margin,
    ).mean()
    residual_l2 = torch.mean((candidate_logits - teacher_logits).square())
    components = {
        "class_balanced_cross_entropy": base,
        "teacher_error_cross_entropy": error_focus,
        "teacher_correct_preservation": preservation,
        "true_role_margin_floor": margin_floor,
        "residual_l2": residual_l2,
    }
    total = (
        float(config["role_cross_entropy_weight"]) * base
        + float(config["teacher_error_cross_entropy_weight"]) * error_focus
        + float(config["teacher_correct_preservation_weight"]) * preservation
        + float(config["true_role_margin_weight"]) * margin_floor
        + float(config["residual_l2_weight"]) * residual_l2
    )
    return total, components


def _load_p1_state() -> dict[str, torch.Tensor]:
    payload = torch.load(
        REPO_ROOT / P1_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError("OCR V28 P1 checkpoint state is missing")
    return state


def _positive_probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent[:, 1] / np.sum(exponent, axis=1)


def _p1_trigger_is_terminal(result: dict[str, Any]) -> bool:
    metrics = result.get("selection_metrics", {})
    per_role = metrics.get("per_role_accuracy", {})
    return bool(
        result.get("candidate_id") == "P1"
        and result.get("candidate_consumed") is True
        and result.get("status") == "failed_selection"
        and result.get("selection_gate_passed") is False
        and result.get("optimizer_steps") == 1024
        and result.get("passing_threshold_window") == []
        and result.get("case_level_details_emitted") is False
        and result.get("public_gate_archive_opened") is False
        and result.get("public_gate_evaluations") == 0
        and metrics.get("scene_count") == 128
        and metrics.get("exact_scene_count") == 124
        and metrics.get("true_positives") == 1024
        and metrics.get("false_positives") == 0
        and metrics.get("false_negatives") == 0
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and metrics.get("role_accuracy") == 0.99609375
        and per_role.get("AxisTitle") == 0.9765625
        and per_role.get("YTick") == 0.9921875
        and result.get("onnx_parity_passed") is True
        and "cases" not in result
        and "predictions" not in result
    )


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-relational-neighborhood-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-p1-relational-role-residual-v1",
        "objective": "class-balanced-role-cross-entropy-plus-parent-preservation-margin-v1",
        "model_license": "Apache-2.0",
        "seed": SEED + 1,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "epochs": 4,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1024,
        "gradient_clip_norm": 5.0,
        "role_cross_entropy_weight": 1.0,
        "teacher_error_cross_entropy_weight": 4.0,
        "teacher_correct_margin_drop_maximum": 0.0,
        "teacher_correct_preservation_weight": 4.0,
        "true_role_logit_margin_floor": 1.0,
        "true_role_margin_weight": 2.0,
        "residual_l2_weight": 0.01,
        "role_residual_scale": 0.25,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 10000,
        "detector_prefilter_applied": False,
        "recognition_batch_size": 64,
        "feature_count": FEATURE_COUNT,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "relation_feature_count": RELATION_FEATURE_COUNT,
        "runtime_numeric_precision": "float32",
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "p1_result_sha256": P1_RESULT_SHA256,
        "p1_checkpoint_path": P1_CHECKPOINT_PATH.as_posix(),
        "p1_checkpoint_sha256": P1_CHECKPOINT_SHA256,
        "p1_onnx_path": P1_ONNX_PATH.as_posix(),
        "p1_onnx_sha256": P1_ONNX_SHA256,
        "p1_report_path": P1_REPORT_PATH.as_posix(),
        "p1_report_sha256": P1_REPORT_SHA256,
        "p1_selection_output_tensor_stream_sha256": P1_SELECTION_OUTPUT_STREAM_SHA256,
        "p2_preregistration_path": PREREGISTRATION_PATH.as_posix(),
        "selection_thresholds": list(THRESHOLDS),
        "split_seal_path": SEAL_PATH.as_posix(),
        "train_fixture_archive_path": TRAIN_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "p1_torch_proposal_logits_exact_required": True,
        "p1_onnx_proposal_logits_exact_required": True,
        "p1_acceptance_at_every_fixed_threshold_exact_required": True,
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
            raise RuntimeError(f"OCR V28 P2 config field mismatch: {key}")
    if config.get("trainable_parameter_names") != list(TRAINABLE_PARAMETER_NAMES):
        raise RuntimeError("OCR V28 P2 trainable parameter boundary changed")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V28 P2 runner source bundle changed")
    if sha256_file(REPO_ROOT / PREREGISTRATION_PATH) != config.get(
        "p2_preregistration_sha256"
    ):
        raise RuntimeError("OCR V28 P2 preregistration changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V28 split seal changed before P2")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V28 split source commit is not an ancestor")
    for relative, expected_hash in seal.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V28 frozen split source changed: {relative}")
    archives = {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (path, key) in archives.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V28 {split} archive changed before P2")
    exact_inputs = {
        P1_RESULT_PATH: P1_RESULT_SHA256,
        P1_CHECKPOINT_PATH: P1_CHECKPOINT_SHA256,
        P1_ONNX_PATH: P1_ONNX_SHA256,
        P1_REPORT_PATH: P1_REPORT_SHA256,
        Path(DETECTOR_PATH): DETECTOR_SHA256,
        Path(RECOGNIZER_PATH): RECOGNIZER_SHA256,
        Path(RECOGNIZER_YAML_PATH): RECOGNIZER_YAML_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V28 P2 frozen input changed: {relative.as_posix()}")
    if not _p1_trigger_is_terminal(_read_json(REPO_ROOT / P1_RESULT_PATH)):
        raise RuntimeError("OCR V28 P2 aggregate-only P1 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V28 P2 output already exists")
    ledger = _read_json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next((
        item for item in ledger.get("revisions", [])
        if item.get("task") == TASK and item.get("revision") == REVISION
    ), None)
    if (
        entry is None
        or entry.get("status") != "candidate_2_preregistered"
        or entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != ["P1"]
    ):
        raise RuntimeError("OCR V28 P2 canonical authorization changed")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V28 P2 output exists: {output_dir}")
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
            return np.asarray(detector_session.run(None, {
                detector_input: np.ascontiguousarray(values),
            })[0], dtype=np.float32)

        def recognizer_runner(values: np.ndarray) -> np.ndarray:
            return np.asarray(recognizer_session.run(None, {
                recognizer_input: np.ascontiguousarray(values),
            })[0], dtype=np.float32)

        phase = "direct_training_feature_crop_and_relation_execution"
        (
            train_values,
            train_crops,
            train_labels,
            train_records,
            train_relations,
            train_slices,
            training_evidence,
        ) = extract_relational_evidence(
            train_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        if train_values.shape[1:] != (FEATURE_COUNT,):
            raise RuntimeError("OCR V28 P2 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V28 P2 training crop shape changed")
        if len(train_relations) != len(train_scenes) or len(train_slices) != len(train_scenes):
            raise RuntimeError("OCR V28 P2 training scene relation stream changed")
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V28 P2 incomplete training stream: {key}")
        train_roles = _role_targets(train_scenes, train_records)
        if int(train_labels.sum()) != sum(len(scene.truths) for scene in train_scenes):
            raise RuntimeError("OCR V28 P2 training truth stream is incomplete")
        if np.any(train_roles[train_labels == 1] < 0):
            raise RuntimeError("OCR V28 P2 positive role target is absent")

        phase = "frozen_p1_relational_role_residual_training"
        generator = _configure(int(config["seed"]))
        model = FrozenP1RelationalRoleResidualNet(
            residual_seed=int(config["seed"]),
            residual_scale=float(config["role_residual_scale"]),
        )
        model.load_p1_state_dict(_load_p1_state())
        frozen_before = _parameter_stream_sha256(model.p1)
        trainable_before = _module_parameter_stream_sha256(model.role_residual)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name.removeprefix("role_residual.")
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        trainable_names = [f"role_residual.{name}" for name in trainable_names]
        if tuple(trainable_names) != TRAINABLE_PARAMETER_NAMES:
            raise RuntimeError("OCR V28 P2 trainable parameter boundary changed")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        role_weights = _balanced_class_weights(
            torch.from_numpy(train_roles[train_labels == 1]),
            len(ROLE_ORDER),
            "role",
        )
        loss_checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            sums = {
                "total": 0.0,
                "class_balanced_cross_entropy": 0.0,
                "teacher_error_cross_entropy": 0.0,
                "teacher_correct_preservation": 0.0,
                "true_role_margin_floor": 0.0,
                "residual_l2": 0.0,
            }
            for scene_index in torch.randperm(
                len(train_slices), generator=generator,
            ).tolist():
                scene_slice = train_slices[scene_index]
                values = torch.from_numpy(train_values[scene_slice]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[scene_slice]).unsqueeze(0)
                relations = torch.from_numpy(train_relations[scene_index]).unsqueeze(0)
                labels = torch.from_numpy(train_labels[scene_slice])
                roles = torch.from_numpy(train_roles[scene_slice])
                positive = labels == 1
                candidate = model(values, crops, relations)[0, positive, 2:]
                with torch.no_grad():
                    teacher = model.p1(values, crops, relations)[0, positive, 2:]
                loss, components = _role_residual_objective(
                    candidate, teacher, roles[positive], role_weights, config,
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
            count = len(train_slices)
            loss_checkpoints.append({
                "epoch": epoch + 1,
                **{name: value / count for name, value in sums.items()},
            })
            print(
                f"OCR V28 P2 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V28 P2 optimizer-step count changed")
        frozen_after = _parameter_stream_sha256(model.p1)
        trainable_after = _module_parameter_stream_sha256(model.role_residual)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V28 P2 modified frozen P1 parameters")
        if trainable_after == trainable_before:
            raise RuntimeError("OCR V28 P2 role residual did not change")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-relational-neighborhood-proposal-v28-p2.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-relational-neighborhood-proposal-v28-p2.onnx"
        first = train_slices[0]
        model.eval()
        _export(
            model,
            torch.from_numpy(train_values[first]).unsqueeze(0),
            torch.from_numpy(train_crops[first]).unsqueeze(0),
            torch.from_numpy(train_relations[0]).unsqueeze(0),
            onnx_path,
        )
        candidate_session = _candidate_session(onnx_path)
        p1_session = _candidate_session(REPO_ROOT / P1_ONNX_PATH)

        phase = "single_visible_selection"
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        _validate_stored_split(
            selection_scenes, evidence["seal"]["splits"]["validation"], "validation",
        )
        (
            selection_values,
            selection_crops,
            _,
            selection_records,
            selection_relations,
            selection_slices,
            selection_evidence,
        ) = extract_relational_evidence(
            selection_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V28 P2 incomplete selection stream: {key}")

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        relation_inputs = sha256()
        candidate_outputs = sha256()
        p1_outputs = sha256()
        p1_proposals = sha256()
        p2_proposals = sha256()
        parity_error = 0.0
        p1_reexecution_parity_error = 0.0
        torch_proposal_mismatches = 0
        onnx_proposal_mismatches = 0
        threshold_acceptance_mismatches = 0
        with torch.inference_mode():
            for scene_index, scene_slice in enumerate(selection_slices):
                values = np.ascontiguousarray(selection_values[scene_slice][None, ...])
                crops = np.ascontiguousarray(selection_crops[scene_slice][None, ...])
                relations = np.ascontiguousarray(selection_relations[scene_index][None, ...])
                torch_values = torch.from_numpy(values)
                torch_crops = torch.from_numpy(crops)
                torch_relations = torch.from_numpy(relations)
                expected = model(torch_values, torch_crops, torch_relations).numpy()
                expected_p1 = model.p1(
                    torch_values, torch_crops, torch_relations,
                ).numpy()
                torch_proposal_mismatches += int(np.count_nonzero(
                    expected[:, :, :2] != expected_p1[:, :, :2]
                ))
                actual = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                    "proposal_relations": relations,
                })[0], dtype=np.float32)
                actual_p1 = np.asarray(p1_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                    "proposal_relations": relations,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected - actual))),
                )
                p1_reexecution_parity_error = max(
                    p1_reexecution_parity_error,
                    float(np.max(np.abs(expected_p1 - actual_p1))),
                )
                onnx_proposal_mismatches += int(np.count_nonzero(
                    actual[:, :, :2] != actual_p1[:, :, :2]
                ))
                p2_probability = _positive_probabilities(actual[0, :, :2])
                p1_probability = _positive_probabilities(actual_p1[0, :, :2])
                for threshold in THRESHOLDS:
                    threshold_acceptance_mismatches += int(np.count_nonzero(
                        (p2_probability >= threshold) != (p1_probability >= threshold)
                    ))
                outputs.append(actual[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                relation_inputs.update(relations.tobytes(order="C"))
                candidate_outputs.update(actual.tobytes(order="C"))
                p1_outputs.update(actual_p1.tobytes(order="C"))
                p1_proposals.update(np.ascontiguousarray(actual_p1[:, :, :2]).tobytes(order="C"))
                p2_proposals.update(np.ascontiguousarray(actual[:, :, :2]).tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        p1_output_hash = p1_outputs.hexdigest()
        p1_proposal_hash = p1_proposals.hexdigest()
        p2_proposal_hash = p2_proposals.hexdigest()
        p1_stream_preserved = p1_output_hash == P1_SELECTION_OUTPUT_STREAM_SHA256
        proposal_preserved = bool(
            torch_proposal_mismatches == 0
            and onnx_proposal_mismatches == 0
            and threshold_acceptance_mismatches == 0
            and p1_proposal_hash == p2_proposal_hash
            and p1_stream_preserved
        )
        selection_evidence.update({
            "candidate_inference_calls": len(selection_slices),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_relation_input_tensor_stream_sha256": relation_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "p1_full_output_tensor_stream_sha256": p1_output_hash,
            "p1_proposal_output_tensor_stream_sha256": p1_proposal_hash,
            "p2_proposal_output_tensor_stream_sha256": p2_proposal_hash,
            "p1_torch_proposal_logit_mismatch_count": torch_proposal_mismatches,
            "p1_onnx_proposal_logit_mismatch_count": onnx_proposal_mismatches,
            "p1_threshold_acceptance_mismatch_count": threshold_acceptance_mismatches,
            "p1_full_output_stream_preserved": p1_stream_preserved,
            "p1_proposal_decisions_preserved": proposal_preserved,
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
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
                item["metrics"]["role_accuracy"],
            ),
        )
        window = robust[1] if robust else ()
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        p1_parity_passed = p1_reexecution_parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        frozen_preserved = frozen_before == frozen_after
        passed = bool(
            robust is not None
            and parity_passed
            and p1_parity_passed
            and frozen_preserved
            and proposal_preserved
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-relational-neighborhood-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": trainable_after != trainable_before,
            "objective": config["objective"],
            "training_evidence": training_evidence,
            "loss_checkpoints": loss_checkpoints,
            "trainable_parameter_names": trainable_names,
            "frozen_p1_parameter_stream_sha256_before": frozen_before,
            "frozen_p1_parameter_stream_sha256_after": frozen_after,
            "frozen_p1_parameters_preserved": frozen_preserved,
            "role_residual_parameter_stream_sha256_before": trainable_before,
            "role_residual_parameter_stream_sha256_after": trainable_after,
            "p1_checkpoint_sha256": P1_CHECKPOINT_SHA256,
            "p1_onnx_sha256": P1_ONNX_SHA256,
            "p1_result_sha256": P1_RESULT_SHA256,
            "p1_report_sha256": P1_REPORT_SHA256,
            "p1_reexecution_parity_maximum_absolute_error": p1_reexecution_parity_error,
            "p1_reexecution_parity_passed": p1_parity_passed,
            "p1_proposal_decisions_preserved": proposal_preserved,
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
            "p2_preregistration_sha256": config["p2_preregistration_sha256"],
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
            "schema": "graphreader.ocr-relational-neighborhood-failure.v1",
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
        "p1_proposal_decisions_preserved": report[
            "p1_proposal_decisions_preserved"
        ],
        "onnx_parity_maximum_absolute_error": report[
            "onnx_parity_maximum_absolute_error"
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
    "TRAINABLE_PARAMETER_NAMES",
    "_p1_trigger_is_terminal",
    "_role_residual_objective",
    "preflight",
    "train_candidate",
]
