# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use morphology-veto proposal training for OCR V27 P1."""

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
    _feature_groups,
    _is_ancestor,
    _read_json,
    _repository_head,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive, proposal_summary, split_fingerprint
from .features import structure_features
from .model import FrozenV26MorphologyVetoNet
from .pipeline import extract_crop_evidence
from .protocol import (
    CROP_CHANNELS,
    CROP_HEIGHT,
    CROP_WIDTH,
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    OUTPUT_LOGIT_SCALE,
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
    STRUCTURE_FEATURE_COUNT,
    TASK,
    THRESHOLDS,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/morphology_veto_proposal_v27")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v27-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v27-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v27-public.zip")
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "features.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    Path(TRIGGER_RESULT_PATH),
    Path("ml/ocr/scene_topology_proposal_v26/dataset.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model_p2.py"),
    Path("ml/ocr/scene_topology_proposal_v26/model_p3.py"),
    Path("ml/ocr/scene_topology_proposal_v26/pipeline.py"),
    Path("ml/ocr/scene_topology_proposal_v26/protocol.py"),
    Path("ml/ocr/evidence_rescue_v25/model.py"),
    Path("ml/ocr/evidence_rescue_v25/train_p1.py"),
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
        digest.update(
            np.ascontiguousarray(parameter.detach().numpy()).tobytes(order="C")
        )
    return digest.hexdigest()


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V27 {name} stored fixtures violate the seal")


def _load_parent_state() -> dict[str, torch.Tensor]:
    payload = torch.load(
        REPO_ROOT / PARENT_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("OCR V27 parent checkpoint state is missing")
    return state


def _candidate_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V27 requires CPUExecutionProvider only")
    return session


def _export(
    model: nn.Module,
    evidence: torch.Tensor,
    crops: torch.Tensor,
    structure: torch.Tensor,
    path: Path,
) -> None:
    torch.onnx.export(
        model,
        (evidence, crops, structure),
        path,
        input_names=["proposal_evidence", "proposal_crops", "structure_features"],
        output_names=["proposal_and_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "structure_features": {1: "proposal_count"},
            "proposal_and_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def _proposal_objective(
    logits: torch.Tensor,
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
        raise RuntimeError("OCR V27 requires positive and negative proposals per scene")
    asymmetric = torch.where(
        negative,
        per_example * float(config["false_positive_weight"]),
        per_example,
    ).mean()
    margin = logits[:, 1] - logits[:, 0]
    positive_margins = margin[positive]
    negative_margins = margin[negative]
    positive_floor = torch.relu(
        float(config["positive_logit_margin_floor"]) - positive_margins
    ).max()
    negative_ceiling = torch.relu(
        negative_margins - float(config["negative_logit_margin_ceiling"])
    ).max()
    scene_separation = torch.relu(
        negative_margins.max()
        - positive_margins.min()
        + float(config["scene_separation_logit_margin_minimum"])
    )
    losses = {
        "asymmetric_cross_entropy": asymmetric,
        "positive_floor": positive_floor,
        "negative_ceiling": negative_ceiling,
        "scene_separation": scene_separation,
    }
    total = (
        float(config["proposal_cross_entropy_weight"]) * asymmetric
        + float(config["positive_floor_weight"]) * positive_floor
        + float(config["negative_ceiling_weight"]) * negative_ceiling
        + float(config["scene_separation_weight"]) * scene_separation
    )
    return total, losses


def _trigger_is_terminal(trigger: dict[str, Any]) -> bool:
    metrics = trigger.get("selection_metrics", {})
    return bool(
        trigger.get("status") == "failed_selection"
        and trigger.get("candidate_id") == "P3"
        and trigger.get("candidate_consumed") is True
        and trigger.get("case_level_details_emitted") is False
        and metrics.get("scene_count") == 128
        and metrics.get("exact_scene_count") == 122
        and metrics.get("true_positives") == 1024
        and metrics.get("false_positives") == 1
        and metrics.get("false_negatives") == 0
        and metrics.get("prohibited_structure_hits") == 1
        and metrics.get("duplicate_region_count") == 0
        and trigger.get("onnx_parity_passed") is False
        and trigger.get("public_gate_archive_opened") is False
        and trigger.get("public_gate_evaluations") == 0
        and "cases" not in trigger
        and "predictions" not in trigger
    )


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-morphology-veto-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-v26-morphology-veto-scaled-v1",
        "objective": "morphology-veto-asymmetric-scene-margin-v1",
        "model_license": "Apache-2.0",
        "seed": SEED,
        "learning_rate": 0.0005,
        "weight_decay": 0.0005,
        "epochs": 4,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1024,
        "gradient_clip_norm": 5.0,
        "proposal_cross_entropy_weight": 1.0,
        "false_positive_weight": 4.0,
        "positive_logit_margin_floor": 2.0,
        "negative_logit_margin_ceiling": -2.0,
        "positive_floor_weight": 2.0,
        "negative_ceiling_weight": 4.0,
        "scene_separation_logit_margin_minimum": 4.0,
        "scene_separation_weight": 2.0,
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
        "parent_checkpoint_path": PARENT_CHECKPOINT_PATH,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_onnx_path": PARENT_ONNX_PATH,
        "parent_onnx_sha256": PARENT_ONNX_SHA256,
        "trigger_result_path": TRIGGER_RESULT_PATH,
        "trigger_result_sha256": TRIGGER_RESULT_SHA256,
        "selection_thresholds": list(THRESHOLDS),
        "split_seal_path": SEAL_PATH.as_posix(),
        "train_fixture_archive_path": TRAIN_ARCHIVE_PATH.as_posix(),
        "selection_fixture_archive_path": SELECTION_ARCHIVE_PATH.as_posix(),
        "public_fixture_archive_path": PUBLIC_ARCHIVE_PATH.as_posix(),
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
            raise RuntimeError(f"OCR V27 P1 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V27 P1 runner source bundle changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V27 split seal changed before P1")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    required_seal = {
        "schema": "graphreader.ocr-morphology-veto-split-seal.v1",
        "revision": REVISION,
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in required_seal.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V27 split seal field changed: {key}")
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V27 split source commit is not an ancestor")
    if seal.get("source_bundle_sha256") != config.get("split_source_bundle_sha256"):
        raise RuntimeError("OCR V27 split source bundle changed")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V27 split source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V27 frozen split source changed: {relative}")
    archive_bindings = {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (path, key) in archive_bindings.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(key):
            raise RuntimeError(f"OCR V27 {split} archive changed before P1")
    exact_inputs = {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        PARENT_CHECKPOINT_PATH: PARENT_CHECKPOINT_SHA256,
        PARENT_ONNX_PATH: PARENT_ONNX_SHA256,
        TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V27 frozen input changed: {relative}")
    if not _trigger_is_terminal(_read_json(REPO_ROOT / TRIGGER_RESULT_PATH)):
        raise RuntimeError("OCR V27 aggregate-only V26 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V27 P1 output already exists")
    ledger = _read_json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(
        (
            item for item in ledger.get("revisions", [])
            if item.get("task") == TASK and item.get("revision") == REVISION
        ),
        None,
    )
    if (
        entry is None
        or entry.get("status") != "candidate_1_preregistered"
        or entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or entry.get("preregistered_candidate_ids") != [CANDIDATE_ID]
        or entry.get("consumed_candidate_ids") != []
    ):
        raise RuntimeError("OCR V27 P1 canonical authorization changed")
    return {"config": config, "head": head, "seal": seal}


def _attach_structure(
    crops: np.ndarray, evidence: dict[str, Any], prefix: str,
) -> np.ndarray:
    values = structure_features(crops)
    if values.shape != (len(crops), STRUCTURE_FEATURE_COUNT) or values.dtype != np.float32:
        raise RuntimeError(f"OCR V27 {prefix} structure contract changed")
    evidence[f"{prefix}_structure_tensor_shape"] = list(values.shape)
    evidence[f"{prefix}_structure_tensor_stream_sha256"] = sha256(
        values.tobytes(order="C")
    ).hexdigest()
    return values


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V27 P1 output exists: {output_dir}")
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

        phase = "direct_training_feature_crop_and_structure_execution"
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
            raise RuntimeError("OCR V27 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V27 training crop shape changed")
        train_structure = _attach_structure(train_crops, training_evidence, "training")
        train_groups = _feature_groups(train_records, len(train_scenes))
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V27 incomplete training stream: {key}")

        phase = "morphology_veto_proposal_training"
        generator = _configure(int(config["seed"]))
        model = FrozenV26MorphologyVetoNet(seed=int(config["seed"]))
        model.load_parent_state_dict(_load_parent_state())
        frozen_before = _parameter_stream_sha256(model.parent)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if not trainable or any(name.startswith("parent.") for name in trainable_names):
            raise RuntimeError("OCR V27 frozen parent boundary changed")
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
        for epoch in range(int(config["epochs"])):
            sums = {
                "total": 0.0,
                "asymmetric_cross_entropy": 0.0,
                "positive_floor": 0.0,
                "negative_ceiling": 0.0,
                "scene_separation": 0.0,
            }
            for scene_index in torch.randperm(
                len(train_groups), generator=generator,
            ).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[indices]).unsqueeze(0)
                structure = torch.from_numpy(train_structure[indices]).unsqueeze(0)
                targets = torch.from_numpy(train_labels[indices])
                candidate = model(values, crops, structure)[0, :, :2]
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
                f"OCR V27 P1 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V27 optimizer-step count changed")
        frozen_after = _parameter_stream_sha256(model.parent)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V27 modified the frozen V26 parent")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-morphology-veto-proposal-v27-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-morphology-veto-proposal-v27-p1.onnx"
        first = train_groups[0]
        example_values = torch.from_numpy(train_values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(train_crops[first]).unsqueeze(0)
        example_structure = torch.from_numpy(train_structure[first]).unsqueeze(0)
        model.eval()
        _export(model, example_values, example_crops, example_structure, onnx_path)
        candidate_session = _candidate_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops", "structure_features",
        }:
            raise RuntimeError("OCR V27 ONNX input identity changed")

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
        selection_structure = _attach_structure(
            selection_crops, selection_evidence, "selection",
        )
        selection_groups = _feature_groups(selection_records, len(selection_scenes))
        registered = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V27 incomplete selection stream: {key}")

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        structure_inputs = sha256()
        candidate_outputs = sha256()
        parent_outputs = sha256()
        parity_error = 0.0
        role_argmax_mismatches = 0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                structure = np.ascontiguousarray(selection_structure[indices][None, ...])
                torch_values = torch.from_numpy(values)
                torch_crops = torch.from_numpy(crops)
                torch_structure = torch.from_numpy(structure)
                expected_output = model(torch_values, torch_crops, torch_structure).numpy()
                parent_output = model.parent(torch_values, torch_crops).numpy()
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
                    != np.argmax(parent_output[:, :, 2:], axis=2)
                ))
                outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                structure_inputs.update(structure.tobytes(order="C"))
                candidate_outputs.update(actual_output.tobytes(order="C"))
                parent_outputs.update(parent_output.tobytes(order="C"))
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_structure_input_tensor_stream_sha256": structure_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "v26_parent_output_tensor_stream_sha256": parent_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
            "parent_role_argmax_mismatch_count": role_argmax_mismatches,
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
        roles_preserved = role_argmax_mismatches == 0
        frozen_preserved = frozen_before == frozen_after
        passed = robust is not None and parity_passed and roles_preserved and frozen_preserved
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
            "frozen_v26_parent_preserved": frozen_preserved,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_role_argmax_mismatch_count": role_argmax_mismatches,
            "parent_role_argmax_preserved": roles_preserved,
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
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
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
        "onnx_parity_maximum_absolute_error": report[
            "onnx_parity_maximum_absolute_error"
        ],
        "parent_role_argmax_mismatch_count": report[
            "parent_role_argmax_mismatch_count"
        ],
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT", "CONFIG_PATH", "RUNNER_SOURCE_PATHS",
    "_proposal_objective", "preflight", "train_candidate",
]
