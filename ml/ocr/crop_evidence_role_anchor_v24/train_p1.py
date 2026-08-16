# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use CPU training and visible selection for OCR V24 P1."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.margin_calibrator_v20.pipeline import (
    ProposalRecord, evaluate_thresholds, select_robust_window,
)
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_archive, proposal_summary, split_fingerprint
from .model import CropEvidenceRoleAnchorNet
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
    V23_RESULT_PATH,
    V23_RESULT_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/crop_evidence_role_anchor_v24")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v24-public.zip")
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p1.py",
    Path("ml/ocr/role_anchor_set_v23/dataset.py"),
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


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"OCR V24 expected a JSON object: {path}")
    return value


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def _proposal_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = nn.functional.cross_entropy(logits, targets, weight=class_weights)
    acceptance_logit = logits[:, 1] - logits[:, 0]
    negative = acceptance_logit[targets == 0]
    positive = acceptance_logit[targets == 1]
    if negative.numel() == 0 or positive.numel() == 0:
        raise RuntimeError("OCR V24 P1 requires positive and negative proposals in every scene")
    negative_limit = _logit(float(config["negative_acceptance_probability_maximum"]))
    positive_limit = _logit(float(config["positive_acceptance_probability_minimum"]))
    negative_margin = torch.relu(negative.max() - negative_limit)
    positive_margin = torch.relu(positive_limit - positive.min())
    total = (
        base
        + float(config["negative_scene_extrema_margin_weight"]) * negative_margin
        + float(config["positive_scene_extrema_margin_weight"]) * positive_margin
    )
    return total, negative_margin, positive_margin


def _balanced_class_weights(
    targets: torch.Tensor, class_count: int, evidence_name: str,
) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=class_count).to(torch.float32)
    if counts.shape != (class_count,) or torch.any(counts == 0):
        raise RuntimeError(f"OCR V24 P1 requires every {evidence_name} class")
    weights = counts.reciprocal()
    weights *= class_count / weights.sum()
    return weights


def _repository_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _is_ancestor(commit: str, head: str) -> bool:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return False
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, head), cwd=REPO_ROOT,
        check=False, capture_output=True,
    )
    return completed.returncode == 0


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V24 requires CPUExecutionProvider only")
    return session


def _configure(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.Generator().manual_seed(seed)


def _export(
    model: nn.Module, evidence: torch.Tensor, crops: torch.Tensor, path: Path,
) -> None:
    torch.onnx.export(
        model,
        (evidence, crops),
        path,
        input_names=["proposal_evidence", "proposal_crops"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )


def _feature_groups(
    records: tuple[ProposalRecord, ...], scene_count: int,
) -> tuple[np.ndarray, ...]:
    groups = tuple(
        np.asarray(
            [index for index, record in enumerate(records) if record.scene_index == scene_index],
            dtype=np.int64,
        )
        for scene_index in range(scene_count)
    )
    if any(len(indices) == 0 for indices in groups):
        raise RuntimeError("OCR V24 production proposal stream omitted an entire scene")
    if not np.array_equal(np.concatenate(groups), np.arange(len(records), dtype=np.int64)):
        raise RuntimeError("OCR V24 production proposal stream is not scene-contiguous")
    return groups


def _role_targets(
    scenes: tuple[Any, ...], records: tuple[ProposalRecord, ...],
) -> np.ndarray:
    indices = {role: index for index, role in enumerate(ROLE_ORDER)}
    return np.asarray([
        -100 if record.truth_index < 0
        else indices[scenes[record.scene_index].truths[record.truth_index].role]
        for record in records
    ], dtype=np.int64)


def _calibrated_records(
    records: tuple[ProposalRecord, ...], outputs: np.ndarray,
) -> tuple[ProposalRecord, ...]:
    if outputs.shape != (len(records), 2 + len(ROLE_ORDER)):
        raise RuntimeError("OCR V24 candidate output stream shape changed")
    return tuple(
        replace(record, predicted_role=ROLE_ORDER[int(np.argmax(output[2:]))])
        for record, output in zip(records, outputs, strict=True)
    )


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-crop-evidence-role-anchor-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "crop-evidence-role-conditioned-scene-anchor-set-v1",
        "objective": "class_balanced_cross_entropy_plus_scene_extrema_acceptance_margin_v1",
        "model_license": "Apache-2.0",
        "seed": SEED,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "negative_acceptance_probability_maximum": 0.10,
        "negative_scene_extrema_margin_weight": 1.5,
        "positive_acceptance_probability_minimum": 0.90,
        "positive_scene_extrema_margin_weight": 0.25,
        "role_loss_weight": 0.75,
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
        "trigger_result_path": V23_RESULT_PATH,
        "trigger_result_sha256": V23_RESULT_SHA256,
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
            raise RuntimeError(f"OCR V24 P1 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V24 P1 thresholds changed")
    if int(config["expected_optimizer_steps"]) > 1280:
        raise RuntimeError("OCR V24 P1 optimizer budget exceeds the preregistration")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V24 P1 runner source bundle changed")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V24 split seal changed before P1")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    seal_expected = {
        "schema": "graphreader.ocr-crop-evidence-role-anchor-split-seal.v1",
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
    for key, value in seal_expected.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V24 split seal field changed: {key}")
    head = _repository_head()
    if not _is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V24 split seal source commit is not an ancestor")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V24 split seal source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
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
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(config_key):
            raise RuntimeError(f"OCR V24 {split} archive changed before P1")
    exact_inputs = {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V24 frozen model input changed: {relative}")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V24 P1 output already exists")
    return {"config": config, "head": head, "seal": seal}


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V24 {name} stored fixtures violate the seal")


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V24 P1 output exists: {output_dir}")
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
            raise RuntimeError("OCR V24 P1 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V24 P1 training crop shape changed")
        train_groups = _feature_groups(train_records, len(train_scenes))
        train_roles = _role_targets(train_scenes, train_records)
        registered_training = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered_training[key]:
                raise RuntimeError(f"OCR V24 P1 incomplete training proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in train_scenes)
        if int(train_labels.sum()) != expected_truths or np.any(train_roles[train_labels == 1] < 0):
            raise RuntimeError("OCR V24 P1 production stream omitted a training truth")

        phase = "crop_evidence_role_anchor_training"
        generator = _configure(int(config["seed"]))
        model = CropEvidenceRoleAnchorNet(seed=int(config["seed"]))
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        proposal_weights = _balanced_class_weights(
            torch.from_numpy(train_labels), 2, "proposal",
        )
        role_weights = _balanced_class_weights(
            torch.from_numpy(train_roles[train_labels == 1]), len(ROLE_ORDER), "role",
        )
        losses: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            epoch_loss = 0.0
            epoch_negative_margin = 0.0
            epoch_positive_margin = 0.0
            for scene_index in torch.randperm(len(train_groups), generator=generator).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[indices]).unsqueeze(0)
                proposal_targets = torch.from_numpy(train_labels[indices])
                role_targets = torch.from_numpy(train_roles[indices])
                output = model(values, crops)[0]
                proposal_loss, negative_margin, positive_margin = _proposal_objective(
                    output[:, :2], proposal_targets, proposal_weights, config,
                )
                positive = proposal_targets == 1
                role_loss = nn.functional.cross_entropy(
                    output[positive, 2:], role_targets[positive], weight=role_weights,
                )
                loss = proposal_loss + float(config["role_loss_weight"]) * role_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
                epoch_negative_margin += float(negative_margin.detach())
                epoch_positive_margin += float(positive_margin.detach())
            count = len(train_groups)
            losses.append({
                "epoch": epoch + 1,
                "loss": epoch_loss / count,
                "negative_scene_extrema_margin": epoch_negative_margin / count,
                "positive_scene_extrema_margin": epoch_positive_margin / count,
            })
            print(
                f"OCR V24 P1 epoch {epoch + 1}/{config['epochs']} complete; "
                f"optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V24 P1 optimizer-step count changed")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-crop-evidence-role-anchor-v24-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-crop-evidence-role-anchor-v24-p1.onnx"
        first = train_groups[0]
        example_values = torch.from_numpy(train_values[first]).unsqueeze(0)
        example_crops = torch.from_numpy(train_crops[first]).unsqueeze(0)
        model.eval()
        _export(model, example_values, example_crops, onnx_path)
        candidate_session = _cpu_session(onnx_path)
        candidate_inputs = {item.name for item in candidate_session.get_inputs()}
        if candidate_inputs != {"proposal_evidence", "proposal_crops"}:
            raise RuntimeError("OCR V24 P1 ONNX input identity changed")

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
        registered_selection = evidence["seal"]["splits"]["validation"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if selection_evidence[key] != registered_selection[key]:
                raise RuntimeError(f"OCR V24 P1 incomplete selection proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in selection_scenes)
        if sum(record.truth_index >= 0 for record in selection_records) != expected_truths:
            raise RuntimeError("OCR V24 P1 production stream omitted a validation truth")
        onnx_outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        candidate_outputs = sha256()
        parity_error = 0.0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                crops = np.ascontiguousarray(selection_crops[indices][None, ...])
                expected_output = model(
                    torch.from_numpy(values), torch.from_numpy(crops),
                ).numpy()
                actual_output = np.asarray(candidate_session.run(None, {
                    "proposal_evidence": values,
                    "proposal_crops": crops,
                })[0], dtype=np.float32)
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
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
        passed = robust is not None and parity_passed
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
                    "negative_acceptance_probability_maximum",
                    "negative_scene_extrema_margin_weight",
                    "positive_acceptance_probability_minimum",
                    "positive_scene_extrema_margin_weight",
                    "role_loss_weight",
                )
            },
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
        "selection_metrics": report["selection_metrics"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT",
    "CONFIG_PATH",
    "RUNNER_SOURCE_PATHS",
    "_balanced_class_weights",
    "_calibrated_records",
    "_feature_groups",
    "_proposal_objective",
    "preflight",
    "train_candidate",
]
