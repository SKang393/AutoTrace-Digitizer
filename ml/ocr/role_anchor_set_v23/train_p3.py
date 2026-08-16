# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use CPU proposal-head repair and visible selection for OCR V23 P3."""

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
from ml.ocr.margin_calibrator_v20.pipeline import (
    evaluate_thresholds,
    extract_features,
    select_robust_window,
)
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_archive, proposal_summary, split_fingerprint
from .model import RoleAnchorSetNet
from .protocol import (
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
    TASK,
    THRESHOLDS,
)
from . import train_p1 as common


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/role_anchor_set_v23")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
PROTOCOL_PATH = ROOT / "P3_PROTOCOL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P1_RESULT_SHA256 = "240a9b6c07a9ace105f5bae780bc2b8966aa8fb432e54d265b63352350b00be5"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_RESULT_SHA256 = "652a5de934933b32757aae557fab43146d23d485620e36a75f7c5e906d0c1e7d"
P1_CHECKPOINT_PATH = ROOT / "artifacts/P1-run/graph-text-role-anchor-set-v23-p1.pt"
P1_CHECKPOINT_SHA256 = "632119b85958de6d0d2db29d19e518c62e2651bd876f129c06aa3846214e26a2"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v23-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v23-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v23-public.zip")
RUNNER_SOURCE_PATHS = tuple(dict.fromkeys((
    PROTOCOL_PATH,
    ROOT / "train_p3.py",
    *common.RUNNER_SOURCE_PATHS,
)))
TRAINABLE_PARAMETER_NAMES = (
    "proposal_head.0.bias",
    "proposal_head.0.weight",
    "proposal_head.2.bias",
    "proposal_head.2.weight",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"OCR V23 expected a JSON object: {path}")
    return value


def _parameter_stream_sha256(
    model: nn.Module,
    *,
    include_trainable: bool,
) -> str:
    digest = sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad is not include_trainable:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(parameter.detach().numpy()).tobytes(order="C"))
    return digest.hexdigest()


def _proposal_head_objective(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the fixed relative teacher and scene-separation margins."""

    base = nn.functional.cross_entropy(logits, targets, weight=class_weights)
    acceptance = logits[:, 1] - logits[:, 0]
    teacher_acceptance = teacher_logits[:, 1] - teacher_logits[:, 0]
    negative = targets == 0
    positive = targets == 1
    if not torch.any(negative) or not torch.any(positive):
        raise RuntimeError("OCR V23 P3 requires positive and negative proposals in every scene")
    allowed_drop = float(config["teacher_positive_logit_drop_maximum"])
    negative_improvement = float(config["teacher_negative_logit_improvement_minimum"])
    separation = float(config["scene_separation_logit_margin_minimum"])
    positive_margin = torch.relu(
        teacher_acceptance[positive] - allowed_drop - acceptance[positive]
    ).max()
    negative_margin = torch.relu(
        acceptance[negative] - (teacher_acceptance[negative] - negative_improvement)
    ).max()
    separation_margin = torch.relu(
        acceptance[negative].max() - acceptance[positive].min() + separation
    )
    total = (
        float(config["proposal_cross_entropy_weight"]) * base
        + float(config["teacher_positive_margin_weight"]) * positive_margin
        + float(config["teacher_negative_margin_weight"]) * negative_margin
        + float(config["scene_separation_margin_weight"]) * separation_margin
    )
    return total, positive_margin, negative_margin, separation_margin


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-role-anchor-set-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "role-conditioned-scene-anchor-set-v1",
        "objective": "class_balanced_cross_entropy_plus_teacher_signed_margin_and_scene_separation_v1",
        "model_license": "Apache-2.0",
        "parent_checkpoint_path": P1_CHECKPOINT_PATH.as_posix(),
        "parent_checkpoint_sha256": P1_CHECKPOINT_SHA256,
        "p1_result_path": P1_RESULT_PATH.as_posix(),
        "p1_result_sha256": P1_RESULT_SHA256,
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_result_sha256": P2_RESULT_SHA256,
        "seed": 2_608_162_303,
        "learning_rate": 0.00005,
        "weight_decay": 0.0,
        "proposal_head_only": True,
        "proposal_cross_entropy_weight": 1.0,
        "teacher_positive_logit_drop_maximum": 0.0,
        "teacher_positive_margin_weight": 2.0,
        "teacher_negative_logit_improvement_minimum": 1.0,
        "teacher_negative_margin_weight": 1.0,
        "scene_separation_logit_margin_minimum": 1.0,
        "scene_separation_margin_weight": 1.5,
        "recognition_batch_size": 64,
        "epochs": 2,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 512,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 100000,
        "detector_prefilter_applied": False,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "validation_or_public_pixels_used_for_training": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "selection_evaluation_limit": 1,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V23 P3 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V23 P3 thresholds changed")
    if config.get("trainable_parameter_names") != list(TRAINABLE_PARAMETER_NAMES):
        raise RuntimeError("OCR V23 P3 trainable parameter boundary changed")
    if int(config["expected_optimizer_steps"]) > 1280:
        raise RuntimeError("OCR V23 P3 optimizer budget exceeds the preregistration")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V23 P3 runner source bundle changed")
    if sha256_file(REPO_ROOT / PROTOCOL_PATH) != config.get("protocol_sha256"):
        raise RuntimeError("OCR V23 P3 protocol changed")
    result_expectations = (
        (P1_RESULT_PATH, P1_RESULT_SHA256, "P1"),
        (P2_RESULT_PATH, P2_RESULT_SHA256, "P2"),
    )
    results: dict[str, dict[str, Any]] = {}
    for path, expected_hash, candidate in result_expectations:
        if sha256_file(REPO_ROOT / path) != expected_hash:
            raise RuntimeError(f"OCR V23 {candidate} aggregate result changed before P3")
        result = _read_json(REPO_ROOT / path)
        for key, value in {
            "candidate_id": candidate,
            "candidate_consumed": True,
            "status": "failed_selection",
            "selection_gate_passed": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "production_approval": False,
            "release_eligible": False,
        }.items():
            if result.get(key) != value:
                raise RuntimeError(f"OCR V23 {candidate} result is not a fail-closed P3 trigger: {key}")
        results[candidate] = result
    if results["P1"]["selection_metrics"]["false_negatives"] != 0:
        raise RuntimeError("OCR V23 P3 requires the aggregate P1 zero-miss parent")
    if results["P2"]["selection_metrics"]["role_accuracy"] >= results["P1"]["selection_metrics"]["role_accuracy"]:
        raise RuntimeError("OCR V23 P3 aggregate trigger no longer records the P2 role regression")
    if sha256_file(REPO_ROOT / P1_CHECKPOINT_PATH) != P1_CHECKPOINT_SHA256:
        raise RuntimeError("OCR V23 P1 parent checkpoint changed before P3")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V23 split seal changed before P3")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    for key, value in {
        "schema": "graphreader.ocr-role-anchor-set-split-seal.v1",
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
    }.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V23 split seal field changed: {key}")
    head = common._repository_head()
    if not common._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V23 split seal source commit is not an ancestor")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V23 split seal source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V23 frozen split source changed: {relative}")
    for split, (path, config_key) in {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(config_key):
            raise RuntimeError(f"OCR V23 {split} archive changed before P3")
    for relative, expected_hash in {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
    }.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V23 frozen model input changed: {relative}")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V23 P3 output already exists")
    return {"config": config, "head": head, "seal": seal, "results": results}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V23 P3 output exists: {output_dir}")
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
        registered = evidence["seal"]["splits"]["train"]
        summary = proposal_summary(train_scenes)
        if (
            split_fingerprint(train_scenes) != registered["split_fingerprint"]
            or any(summary[key] != registered["proposal_summary"][key] for key in summary)
        ):
            raise RuntimeError("OCR V23 train stored fixtures violate the seal")

        detector_session = common._cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = common._cpu_session(REPO_ROOT / RECOGNIZER_PATH)
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

        phase = "direct_training_feature_execution"
        train_values, train_labels, train_records, training_evidence = extract_features(
            train_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        if train_values.shape[1:] != (FEATURE_COUNT,):
            raise RuntimeError("OCR V23 P3 training feature width changed")
        train_groups = common._feature_groups(train_records, len(train_scenes))
        registered_training = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered_training[key]:
                raise RuntimeError(f"OCR V23 P3 incomplete training proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in train_scenes)
        if int(train_labels.sum()) != expected_truths:
            raise RuntimeError("OCR V23 P3 production stream omitted a training truth")

        phase = "proposal_head_only_training"
        generator = common._configure(int(config["seed"]))
        checkpoint = torch.load(REPO_ROOT / P1_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
        if not isinstance(state_dict, dict):
            raise RuntimeError("OCR V23 P1 parent checkpoint has no state_dict")
        teacher = RoleAnchorSetNet()
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        model = RoleAnchorSetNet()
        model.load_state_dict(state_dict, strict=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in TRAINABLE_PARAMETER_NAMES)
        actual_trainable = tuple(sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ))
        if actual_trainable != TRAINABLE_PARAMETER_NAMES:
            raise RuntimeError("OCR V23 P3 proposal-head-only boundary changed")
        frozen_before = _parameter_stream_sha256(model, include_trainable=False)
        optimizer = torch.optim.AdamW(
            model.proposal_head.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        proposal_weights = common._balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        losses: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            epoch_loss = 0.0
            epoch_positive = 0.0
            epoch_negative = 0.0
            epoch_separation = 0.0
            for scene_index in torch.randperm(len(train_groups), generator=generator).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                targets = torch.from_numpy(train_labels[indices])
                with torch.inference_mode():
                    teacher_logits = teacher(values)[0, :, :2]
                logits = model(values)[0, :, :2]
                loss, positive_margin, negative_margin, separation_margin = _proposal_head_objective(
                    logits, teacher_logits, targets, proposal_weights, config,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
                epoch_positive += float(positive_margin.detach())
                epoch_negative += float(negative_margin.detach())
                epoch_separation += float(separation_margin.detach())
            count = len(train_groups)
            losses.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / count,
                "teacher_positive_margin": epoch_positive / count,
                "teacher_negative_margin": epoch_negative / count,
                "scene_separation_margin": epoch_separation / count,
            })
            print(
                f"OCR V23 P3 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V23 P3 optimizer-step count changed")
        frozen_after = _parameter_stream_sha256(model, include_trainable=False)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V23 P3 changed a frozen parent parameter")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-role-anchor-set-v23-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-role-anchor-set-v23-p3.onnx"
        example = torch.from_numpy(train_values[train_groups[0]]).unsqueeze(0)
        model.eval()
        common._export(model, example, onnx_path)
        candidate_session = common._cpu_session(onnx_path)
        candidate_input = candidate_session.get_inputs()[0].name

        phase = "single_visible_selection"
        selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
        registered = evidence["seal"]["splits"]["validation"]
        summary = proposal_summary(selection_scenes)
        if (
            split_fingerprint(selection_scenes) != registered["split_fingerprint"]
            or any(summary[key] != registered["proposal_summary"][key] for key in summary)
        ):
            raise RuntimeError("OCR V23 validation stored fixtures violate the seal")
        selection_values, _, selection_records, selection_evidence = extract_features(
            selection_scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        selection_groups = common._feature_groups(selection_records, len(selection_scenes))
        registered_selection = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered_selection[key]:
                raise RuntimeError(f"OCR V23 P3 incomplete selection proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in selection_scenes)
        if sum(record.truth_index >= 0 for record in selection_records) != expected_truths:
            raise RuntimeError("OCR V23 P3 production stream omitted a validation truth")
        onnx_outputs: list[np.ndarray] = []
        candidate_inputs = sha256()
        candidate_outputs = sha256()
        parity_error = 0.0
        teacher_role_error = 0.0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                tensor = torch.from_numpy(values)
                expected_output = model(tensor).numpy()
                teacher_output = teacher(tensor).numpy()
                actual_output = np.asarray(
                    candidate_session.run(None, {candidate_input: values})[0], dtype=np.float32,
                )
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
                )
                teacher_role_error = max(
                    teacher_role_error,
                    float(np.max(np.abs(expected_output[:, :, 2:] - teacher_output[:, :, 2:]))),
                )
                onnx_outputs.append(actual_output[0])
                candidate_inputs.update(values.tobytes(order="C"))
                candidate_outputs.update(np.ascontiguousarray(actual_output).tobytes(order="C"))
        flat_output = np.concatenate(onnx_outputs)
        calibrated_records = common._calibrated_records(selection_records, flat_output)
        selection_evidence.update({
            "calibrator_inference_calls": len(selection_groups),
            "calibrator_input_tensor_stream_sha256": candidate_inputs.hexdigest(),
            "calibrator_output_tensor_stream_sha256": candidate_outputs.hexdigest(),
            "calibrator_onnx_sha256": sha256_file(onnx_path),
            "p1_teacher_role_maximum_absolute_error": teacher_role_error,
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
        passed = robust is not None and parity_passed and role_preserved
        report: dict[str, object] = {
            "schema": "graphreader.ocr-role-anchor-set-candidate-report.v1",
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
                )
            },
            "p1_result_sha256": P1_RESULT_SHA256,
            "p2_result_sha256": P2_RESULT_SHA256,
            "parent_checkpoint_sha256": P1_CHECKPOINT_SHA256,
            "trainable_parameter_names": list(TRAINABLE_PARAMETER_NAMES),
            "frozen_parameter_stream_sha256_before": frozen_before,
            "frozen_parameter_stream_sha256_after": frozen_after,
            "p1_teacher_role_maximum_absolute_error": teacher_role_error,
            "p1_teacher_role_preserved": role_preserved,
            "training_evidence": training_evidence,
            "loss_checkpoints": losses,
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
            "schema": "graphreader.ocr-role-anchor-set-failure.v1",
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
            "runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
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
        "p1_teacher_role_maximum_absolute_error": report["p1_teacher_role_maximum_absolute_error"],
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
    "_proposal_head_objective",
    "preflight",
    "train_candidate",
]
