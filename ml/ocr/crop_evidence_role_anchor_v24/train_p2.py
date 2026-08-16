# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use frozen-backbone crop-residual training for OCR V24 P2."""

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
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive
from .model_p2 import FrozenRoleAnchorCropResidualNet
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
    ROLE_ORDER,
    SEED,
    TASK,
    THRESHOLDS,
)
from .train_p1 import (
    _balanced_class_weights,
    _calibrated_records,
    _configure,
    _cpu_session,
    _export,
    _feature_groups,
    _is_ancestor,
    _read_json,
    _repository_head,
    _validate_stored_split,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/crop_evidence_role_anchor_v24")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-public.zip")
PARENT_CHECKPOINT_PATH = Path(
    "ml/ocr/role_anchor_set_v23/artifacts/P3-run/"
    "graph-text-role-anchor-set-v23-p3.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "83d7b47a6fa53ea7b5618acb4b0d4bebb9207594967c4e83ad7c1c62c7cc409d"
)
PARENT_RESULT_PATH = Path("ml/ocr/role_anchor_set_v23/P3_RESULT.json")
PARENT_RESULT_SHA256 = (
    "83d7a3be46e082be3550144cb4bb1b0a287ada29fadbdcca231d2e27d7ad7422"
)
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model_p2.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    ROOT / "train_p2.py",
    P1_RESULT_PATH,
    Path("ml/ocr/role_anchor_set_v23/model.py"),
    Path("ml/ocr/role_anchor_set_v23/protocol.py"),
    PARENT_RESULT_PATH,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    base = nn.functional.cross_entropy(
        candidate_logits, targets, weight=class_weights,
    )
    candidate_margin = candidate_logits[:, 1] - candidate_logits[:, 0]
    teacher_margin = teacher_logits[:, 1] - teacher_logits[:, 0]
    positive = targets == 1
    negative = targets == 0
    if not torch.any(positive) or not torch.any(negative):
        raise RuntimeError("OCR V24 P2 requires positive and negative proposals in every scene")
    positive_drop = torch.relu(
        teacher_margin[positive] - candidate_margin[positive]
        - float(config["teacher_positive_logit_drop_maximum"])
    ).max()
    negative_improvement = torch.relu(
        candidate_margin[negative]
        - teacher_margin[negative]
        + float(config["teacher_negative_logit_improvement_minimum"])
    ).max()
    separation = torch.relu(
        candidate_margin[negative].max()
        - candidate_margin[positive].min()
        + float(config["scene_separation_logit_margin_minimum"])
    )
    total = (
        float(config["proposal_cross_entropy_weight"]) * base
        + float(config["teacher_positive_margin_weight"]) * positive_drop
        + float(config["teacher_negative_margin_weight"]) * negative_improvement
        + float(config["scene_separation_margin_weight"]) * separation
    )
    return total, positive_drop, negative_improvement, separation


def _load_parent_state() -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        REPO_ROOT / PARENT_CHECKPOINT_PATH, map_location="cpu", weights_only=True,
    )
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError("OCR V24 P2 parent checkpoint has no state_dict")
    return state_dict


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-crop-evidence-role-anchor-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "frozen-v23-role-anchor-plus-crop-proposal-residual-v1",
        "objective": "teacher-preserving-crop-proposal-residual-v1",
        "model_license": "Apache-2.0",
        "seed": SEED + 1,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "proposal_cross_entropy_weight": 1.0,
        "teacher_positive_logit_drop_maximum": 0.0,
        "teacher_positive_margin_weight": 2.0,
        "teacher_negative_logit_improvement_minimum": 1.0,
        "teacher_negative_margin_weight": 1.0,
        "scene_separation_logit_margin_minimum": 1.0,
        "scene_separation_margin_weight": 1.5,
        "crop_residual_scale": 0.0625,
        "recognition_batch_size": 64,
        "epochs": 5,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1280,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 100000,
        "detector_prefilter_applied": False,
        "crop_channels": CROP_CHANNELS,
        "crop_height": CROP_HEIGHT,
        "crop_width": CROP_WIDTH,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "parent_checkpoint_path": PARENT_CHECKPOINT_PATH.as_posix(),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_result_path": PARENT_RESULT_PATH.as_posix(),
        "parent_result_sha256": PARENT_RESULT_SHA256,
        "split_source_commit": "41702515c1b13a550f04cbfefe1d393a4e2e13e5",
        "split_source_bundle_sha256": "d82ec502bb9ff8f84929bafc14eab752958bc0958a479c99600e92a0ea43e288",
        "onnx_parity_maximum_absolute_error": 1e-5,
        "validation_or_public_pixels_used_for_training": False,
        "selection_evaluation_limit": 1,
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
            raise RuntimeError(f"OCR V24 P2 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V24 P2 thresholds changed")
    if int(config["expected_optimizer_steps"]) > 1280:
        raise RuntimeError("OCR V24 P2 optimizer budget exceeds the preregistration")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V24 P2 runner source bundle changed")
    exact_inputs = {
        SEAL_PATH: config["split_seal_sha256"],
        P1_RESULT_PATH: config["p1_result_sha256"],
        PARENT_CHECKPOINT_PATH: PARENT_CHECKPOINT_SHA256,
        PARENT_RESULT_PATH: PARENT_RESULT_SHA256,
        Path(DETECTOR_PATH): DETECTOR_SHA256,
        Path(RECOGNIZER_PATH): RECOGNIZER_SHA256,
        Path(RECOGNIZER_YAML_PATH): RECOGNIZER_YAML_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V24 P2 frozen input changed: {relative.as_posix()}")
    p1 = _read_json(REPO_ROOT / P1_RESULT_PATH)
    if (
        p1.get("status") != "failed_selection"
        or p1.get("candidate_consumed") is not True
        or p1.get("selection_metrics", {}).get("false_positives") != 0
        or p1.get("selection_metrics", {}).get("false_negatives") != 22
        or p1.get("selection_metrics", {}).get("role_accuracy") != 0.8818359375
        or p1.get("selection_metrics", {}).get("per_role_accuracy", {}).get("PhaseHeading")
        != 0.421875
        or p1.get("onnx_parity_passed") is not False
        or p1.get("public_gate_archive_opened") is not False
        or p1.get("case_level_details_emitted") is not False
    ):
        raise RuntimeError("OCR V24 P2 aggregate-only P1 trigger changed")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    if sha256_file(REPO_ROOT / SEAL_PATH) != config["split_seal_sha256"]:
        raise RuntimeError("OCR V24 split seal changed before P2")
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V24 split seal source commit is not an ancestor")
    for relative, expected_hash in seal.get("source_sha256", {}).items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V24 frozen split source changed: {relative}")
    archive_bindings = {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (path, config_key) in archive_bindings.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config[config_key]:
            raise RuntimeError(f"OCR V24 {split} archive changed before P2")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V24 P2 output already exists")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V24 P2 output exists: {output_dir}")
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
    phase = "load_frozen_fixtures"
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
            raise RuntimeError("OCR V24 P2 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V24 P2 training crop shape changed")
        train_groups = _feature_groups(train_records, len(train_scenes))
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V24 P2 incomplete training proposal stream: {key}")

        phase = "frozen_role_anchor_crop_residual_training"
        generator = _configure(int(config["seed"]))
        model = FrozenRoleAnchorCropResidualNet(
            seed=int(config["seed"]), residual_scale=float(config["crop_residual_scale"]),
        )
        model.load_backbone_state_dict(_load_parent_state())
        model.backbone.eval()
        frozen_before = _parameter_stream_sha256(model.backbone)
        trainable = model.trainable_parameters()
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        if not trainable or any(name.startswith("backbone.") for name in trainable_names):
            raise RuntimeError("OCR V24 P2 frozen-backbone boundary changed")
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        proposal_weights = _balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        losses: list[dict[str, float | int]] = []
        model.train()
        model.backbone.eval()
        for epoch in range(int(config["epochs"])):
            epoch_loss = 0.0
            epoch_positive = 0.0
            epoch_negative = 0.0
            epoch_separation = 0.0
            for scene_index in torch.randperm(len(train_groups), generator=generator).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[indices]).unsqueeze(0)
                targets = torch.from_numpy(train_labels[indices])
                with torch.no_grad():
                    teacher = model.backbone(values)[0, :, :2]
                candidate = model(values, crops)[0, :, :2]
                loss, positive, negative, separation = _proposal_residual_objective(
                    candidate, teacher, targets, proposal_weights, config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
                epoch_positive += float(positive.detach())
                epoch_negative += float(negative.detach())
                epoch_separation += float(separation.detach())
            count = len(train_groups)
            losses.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / count,
                "teacher_positive_margin": epoch_positive / count,
                "teacher_negative_margin": epoch_negative / count,
                "scene_separation_margin": epoch_separation / count,
            })
            print(
                f"OCR V24 P2 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V24 P2 optimizer-step count changed")
        frozen_after = _parameter_stream_sha256(model.backbone)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V24 P2 modified the frozen V23 backbone")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-crop-evidence-role-anchor-v24-p2.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-crop-evidence-role-anchor-v24-p2.onnx"
        first = train_groups[0]
        example_values = torch.from_numpy(train_values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(train_crops[first]).unsqueeze(0)
        model.eval()
        _export(model, example_values, example_crops, onnx_path)
        candidate_session = _cpu_session(onnx_path)
        if {item.name for item in candidate_session.get_inputs()} != {
            "proposal_evidence", "proposal_crops",
        }:
            raise RuntimeError("OCR V24 P2 ONNX input identity changed")

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
                raise RuntimeError(f"OCR V24 P2 incomplete selection proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in selection_scenes)
        if sum(record.truth_index >= 0 for record in selection_records) != expected_truths:
            raise RuntimeError("OCR V24 P2 production stream omitted a validation truth")

        onnx_outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        candidate_outputs = sha256()
        parity_error = 0.0
        teacher_role_error = 0.0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                torch_values = torch.from_numpy(values)
                expected_output = model(torch_values, torch.from_numpy(crops)).numpy()
                teacher_output = model.backbone(torch_values).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                teacher_role_error = max(
                    teacher_role_error,
                    float(np.max(np.abs(expected_output[:, :, 2:] - teacher_output[:, :, 2:]))),
                )
                onnx_outputs.append(actual_output[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual_output).tobytes(order="C"))
        flat_output = np.concatenate(onnx_outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "candidate_inference_calls": len(selection_groups),
            "candidate_evidence_input_tensor_stream_sha256": evidence_inputs.hexdigest(),
            "candidate_crop_input_tensor_stream_sha256": crop_inputs.hexdigest(),
            "candidate_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "candidate_onnx_sha256": sha256_file(onnx_path),
            "parent_role_maximum_absolute_error": teacher_role_error,
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
        role_preserved = teacher_role_error == 0.0
        frozen_preserved = frozen_before == frozen_after
        passed = robust is not None and parity_passed and role_preserved and frozen_preserved
        report: dict[str, object] = {
            "schema": "graphreader.ocr-crop-evidence-role-anchor-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "objective": config["objective"],
            "objective_configuration": {
                key: config[key] for key in (
                    "proposal_cross_entropy_weight",
                    "teacher_positive_logit_drop_maximum",
                    "teacher_positive_margin_weight",
                    "teacher_negative_logit_improvement_minimum",
                    "teacher_negative_margin_weight",
                    "scene_separation_logit_margin_minimum",
                    "scene_separation_margin_weight",
                    "crop_residual_scale",
                )
            },
            "training_evidence": training_evidence,
            "loss_checkpoints": losses,
            "trainable_parameter_names": trainable_names,
            "frozen_parameter_stream_sha256_before": frozen_before,
            "frozen_parameter_stream_sha256_after": frozen_after,
            "frozen_backbone_preserved": frozen_preserved,
            "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "parent_role_maximum_absolute_error": teacher_role_error,
            "parent_roles_preserved": role_preserved,
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
            "training_authorization": authorization.binding,
            "p1_result_sha256": config["p1_result_sha256"],
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
            "schema": "graphreader.ocr-crop-evidence-role-anchor-failure.v1",
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
