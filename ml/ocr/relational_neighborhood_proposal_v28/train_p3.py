# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer geometry role repair for OCR V28 P3."""

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

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _cpu_session,
    _is_ancestor,
    _read_json,
    _repository_head,
    _role_targets,
)
from ml.ocr.margin_calibrator_v20.pipeline import evaluate_thresholds, select_robust_window
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet

from .dataset import load_archive
from .model_p3 import FrozenP1GeometryRolePartitionNet
from .pipeline import extract_relational_evidence
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
    TASK,
    THRESHOLDS,
)
from .train_p1 import (
    _candidate_session,
    _export,
    _parameter_stream_sha256,
    _validate_stored_split,
)
from .train_p2 import (
    P1_CHECKPOINT_PATH,
    P1_CHECKPOINT_SHA256,
    P1_ONNX_PATH,
    P1_ONNX_SHA256,
    P1_REPORT_PATH,
    P1_REPORT_SHA256,
    P1_RESULT_PATH,
    P1_RESULT_SHA256,
    P1_SELECTION_OUTPUT_STREAM_SHA256,
    PUBLIC_ARCHIVE_PATH,
    RUNNER_SOURCE_PATHS as P2_RUNNER_SOURCE_PATHS,
    SEAL_PATH,
    SELECTION_ARCHIVE_PATH,
    TRAIN_ARCHIVE_PATH,
    _load_p1_state,
    _positive_probabilities,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/relational_neighborhood_proposal_v28")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
PREREGISTRATION_PATH = ROOT / "P3_PREREGISTRATION.json"
TRAINING_GEOMETRY_EVIDENCE_PATH = ROOT / "P3_TRAINING_GEOMETRY_EVIDENCE.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_RESULT_SHA256 = "c66beaf6662b91113921e26274de8d3fe1219966d122367de6bf537b6c14c3c2"
P1_PARAMETER_STREAM_SHA256 = "ce3ca190dd74dbe1add3f30f9e4b1114ee2b87b30fdd4fd8bda5ff08f0bdbc90"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = tuple(dict.fromkeys((
    *P2_RUNNER_SOURCE_PATHS,
    P2_RESULT_PATH,
    PREREGISTRATION_PATH,
    TRAINING_GEOMETRY_EVIDENCE_PATH,
    ROOT / "model_p3.py",
    ROOT / "train_p3.py",
)))


def _p2_trigger_is_terminal(result: dict[str, Any]) -> bool:
    metrics = result.get("selection_metrics", {})
    roles = metrics.get("per_role_accuracy", {})
    return bool(
        result.get("status") == "failed_selection"
        and result.get("candidate_id") == "P2"
        and result.get("candidate_consumed") is True
        and result.get("case_level_details_emitted") is False
        and metrics.get("scene_count") == 128
        and metrics.get("exact_scene_count") == 123
        and metrics.get("true_positives") == 1024
        and metrics.get("false_positives") == 0
        and metrics.get("false_negatives") == 0
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and metrics.get("role_accuracy") == 0.9951171875
        and roles.get("AxisTitle") == 0.9765625
        and roles.get("YTick") == 0.9921875
        and roles.get("Annotation") == 0.9921875
        and result.get("p1_proposal_decisions_preserved") is True
        and result.get("p1_full_output_stream_preserved") is True
        and result.get("onnx_parity_passed") is True
        and result.get("public_gate_archive_opened") is False
        and result.get("public_gate_evaluations") == 0
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
        "architecture": "frozen-p1-geometry-role-partition-v1",
        "objective": "zero-optimizer-source-declared-geometry-role-partition-v1",
        "model_license": "Apache-2.0",
        "optimizer_steps": 0,
        "trainable_parameter_names": [],
        "relative_center_x_index": 25,
        "relative_center_y_index": 26,
        "x_left_boundary": 0.0,
        "x_right_boundary": 1.0,
        "y_above_boundary": 0.0,
        "y_below_boundary": 1.0,
        "x_tick_axis_title_y_boundary": 1.15,
        "role_logit_magnitude": 8.0,
        "expected_training_role_matches": 2048,
        "expected_training_role_mismatches": 0,
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
        "selection_thresholds": list(THRESHOLDS),
        "selection_evaluation_limit": 1,
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "p1_torch_proposal_logits_exact_required": True,
        "p1_onnx_proposal_logits_exact_required": True,
        "p1_acceptance_at_every_fixed_threshold_exact_required": True,
        "validation_or_public_pixels_used_for_design": False,
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
            raise RuntimeError(f"OCR V28 P3 configuration changed: {key}")
    required_hashes = {
        "p3_preregistration_sha256": PREREGISTRATION_PATH,
        "training_geometry_evidence_sha256": TRAINING_GEOMETRY_EVIDENCE_PATH,
        "protocol_sha256": ROOT / "PROTOCOL.json",
        "p1_result_sha256": P1_RESULT_PATH,
        "p2_result_sha256": P2_RESULT_PATH,
        "p1_checkpoint_sha256": P1_CHECKPOINT_PATH,
        "p1_onnx_sha256": P1_ONNX_PATH,
        "p1_report_sha256": P1_REPORT_PATH,
        "split_seal_sha256": SEAL_PATH,
        "train_fixture_archive_sha256": TRAIN_ARCHIVE_PATH,
        "selection_fixture_archive_sha256": SELECTION_ARCHIVE_PATH,
        "public_fixture_archive_sha256": PUBLIC_ARCHIVE_PATH,
        "detector_sha256": Path(DETECTOR_PATH),
        "recognizer_sha256": Path(RECOGNIZER_PATH),
        "recognizer_inference_yaml_sha256": Path(RECOGNIZER_YAML_PATH),
    }
    for key, path in required_hashes.items():
        if config.get(key) != sha256_file(REPO_ROOT / path):
            raise RuntimeError(f"OCR V28 P3 prerequisite changed: {key}")
    if config.get("p1_result_sha256") != P1_RESULT_SHA256:
        raise RuntimeError("OCR V28 P3 P1 result binding changed")
    if config.get("p2_result_sha256") != P2_RESULT_SHA256:
        raise RuntimeError("OCR V28 P3 P2 result binding changed")
    if config.get("p1_checkpoint_sha256") != P1_CHECKPOINT_SHA256:
        raise RuntimeError("OCR V28 P3 P1 checkpoint binding changed")
    if config.get("p1_onnx_sha256") != P1_ONNX_SHA256:
        raise RuntimeError("OCR V28 P3 P1 ONNX binding changed")
    if config.get("p1_report_sha256") != P1_REPORT_SHA256:
        raise RuntimeError("OCR V28 P3 P1 report binding changed")

    trigger = _read_json(REPO_ROOT / P2_RESULT_PATH)
    if not _p2_trigger_is_terminal(trigger):
        raise RuntimeError("OCR V28 P3 requires the consumed aggregate P2 failure")
    preregistration = _read_json(REPO_ROOT / PREREGISTRATION_PATH)
    if (
        preregistration.get("candidate_id") != CANDIDATE_ID
        or preregistration.get("optimizer_steps") != 0
        or preregistration.get("training_geometry_partition_mismatches") != 0
        or preregistration.get("execution_authorized") is not False
        or preregistration.get("public_execution_authorized") is not False
    ):
        raise RuntimeError("OCR V28 P3 preregistration changed")
    geometry = _read_json(REPO_ROOT / TRAINING_GEOMETRY_EVIDENCE_PATH)
    if (
        geometry.get("partition_matches") != 2048
        or geometry.get("partition_mismatches") != 0
        or geometry.get("validation_fixture_opened") is not False
        or geometry.get("public_fixture_opened") is not False
        or geometry.get("case_level_details_emitted") is not False
    ):
        raise RuntimeError("OCR V28 P3 training geometry evidence changed")

    seal = _read_json(REPO_ROOT / SEAL_PATH)
    if (
        seal.get("protocol", {}).get("state")
        != "preregistered_before_fixture_freeze_or_candidate_execution"
        or seal.get("public_execution_authorized") is not False
        or seal.get("public_evaluations") != 0
        or seal.get("chandler_used") is not False
        or seal.get("private_data") is not False
        or seal.get("production_approval") is not False
        or seal.get("release_eligible") is not False
    ):
        raise RuntimeError("OCR V28 P3 split seal changed")
    for split, config_key in (
        ("train", "train_fixture_archive_sha256"),
        ("validation", "selection_fixture_archive_sha256"),
        ("sealed_public", "public_fixture_archive_sha256"),
    ):
        if seal["splits"][split]["archive_sha256"] != config[config_key]:
            raise RuntimeError(f"OCR V28 P3 {split} archive binding changed")

    head = _repository_head()
    if not _is_ancestor(str(config.get("runner_source_commit")), head):
        raise RuntimeError("OCR V28 P3 runner source commit is not an ancestor")
    actual_bundle = source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
    if config.get("expected_runner_source_bundle_sha256") != actual_bundle:
        raise RuntimeError("OCR V28 P3 runner source bundle changed")
    ledger = _read_json(
        REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    )
    entry = next(
        (
            item for item in ledger["revisions"]
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
        or entry.get("remaining_unregistered_candidate_ids") != []
        or entry.get("selection_evaluations") != 2
        or entry.get("public_gate_authorized") is not False
        or entry.get("public_gate_evaluations") != 0
        or entry.get("candidate_config_sha256", {}).get(CANDIDATE_ID)
        != sha256_file(REPO_ROOT / CONFIG_PATH)
    ):
        raise RuntimeError("OCR V28 P3 canonical authorization changed")
    return {"config": config, "head": head, "seal": seal}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V28 P3 output exists: {output_dir}")
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
        _validate_stored_split(
            train_scenes, evidence["seal"]["splits"]["train"], "train",
        )
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

        phase = "direct_training_geometry_partition_execution"
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
            raise RuntimeError("OCR V28 P3 training evidence width changed")
        if train_crops.shape[1:] != (CROP_CHANNELS, CROP_HEIGHT, CROP_WIDTH):
            raise RuntimeError("OCR V28 P3 training crop shape changed")
        if len(train_relations) != len(train_scenes) or len(train_slices) != len(train_scenes):
            raise RuntimeError("OCR V28 P3 training scene relation stream changed")
        registered = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered[key]:
                raise RuntimeError(f"OCR V28 P3 incomplete training stream: {key}")
        train_roles = _role_targets(train_scenes, train_records)

        model = FrozenP1GeometryRolePartitionNet()
        model.load_p1_state_dict(_load_p1_state())
        frozen_before = _parameter_stream_sha256(model.p1)
        if frozen_before != P1_PARAMETER_STREAM_SHA256:
            raise RuntimeError("OCR V28 P3 frozen P1 parameter stream changed")
        if model.trainable_parameters():
            raise RuntimeError("OCR V28 P3 unexpectedly exposed trainable parameters")
        model.eval()
        training_role_outputs = sha256()
        training_role_matches = 0
        training_role_total = 0
        with torch.inference_mode():
            for scene_index, scene_slice in enumerate(train_slices):
                values = torch.from_numpy(train_values[scene_slice]).unsqueeze(0)
                crops = torch.from_numpy(train_crops[scene_slice]).unsqueeze(0)
                relations = torch.from_numpy(train_relations[scene_index]).unsqueeze(0)
                output = model(values, crops, relations)[0]
                labels = torch.from_numpy(train_labels[scene_slice])
                roles = torch.from_numpy(train_roles[scene_slice])
                positive = labels == 1
                predicted = output[positive, 2:].argmax(dim=1)
                expected = roles[positive]
                training_role_matches += int((predicted == expected).sum())
                training_role_total += int(positive.sum())
                training_role_outputs.update(
                    np.ascontiguousarray(output[:, 2:].numpy()).tobytes(order="C")
                )
        training_role_mismatches = training_role_total - training_role_matches
        if (
            training_role_matches != int(config["expected_training_role_matches"])
            or training_role_mismatches != int(config["expected_training_role_mismatches"])
        ):
            raise RuntimeError("OCR V28 P3 training geometry partition gate failed")
        frozen_after = _parameter_stream_sha256(model.p1)
        if frozen_after != frozen_before:
            raise RuntimeError("OCR V28 P3 modified frozen P1 parameters")
        training_evidence.update({
            "geometry_partition_role_total": training_role_total,
            "geometry_partition_role_matches": training_role_matches,
            "geometry_partition_role_mismatches": training_role_mismatches,
            "geometry_partition_output_tensor_stream_sha256": training_role_outputs.hexdigest(),
            "optimizer_steps": 0,
        })

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-relational-neighborhood-proposal-v28-p3.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "role_partition": {
                "relative_center_x_index": 25,
                "relative_center_y_index": 26,
                "x_tick_axis_title_y_boundary": 1.15,
                "role_logit_magnitude": 8.0,
            },
        }, checkpoint_path)
        onnx_path = output_dir / "graph-text-relational-neighborhood-proposal-v28-p3.onnx"
        first = train_slices[0]
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
                raise RuntimeError(f"OCR V28 P3 incomplete selection stream: {key}")

        outputs: list[np.ndarray] = []
        evidence_inputs = sha256()
        crop_inputs = sha256()
        relation_inputs = sha256()
        candidate_outputs = sha256()
        p1_outputs = sha256()
        p1_proposals = sha256()
        p3_proposals = sha256()
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
                p3_probability = _positive_probabilities(actual[0, :, :2])
                p1_probability = _positive_probabilities(actual_p1[0, :, :2])
                for threshold in THRESHOLDS:
                    threshold_acceptance_mismatches += int(np.count_nonzero(
                        (p3_probability >= threshold) != (p1_probability >= threshold)
                    ))
                outputs.append(actual[0])
                evidence_inputs.update(values.tobytes(order="C"))
                crop_inputs.update(crops.tobytes(order="C"))
                relation_inputs.update(relations.tobytes(order="C"))
                candidate_outputs.update(actual.tobytes(order="C"))
                p1_outputs.update(actual_p1.tobytes(order="C"))
                p1_proposals.update(
                    np.ascontiguousarray(actual_p1[:, :, :2]).tobytes(order="C")
                )
                p3_proposals.update(
                    np.ascontiguousarray(actual[:, :, :2]).tobytes(order="C")
                )
        flat_output = np.concatenate(outputs)
        calibrated_records = _calibrated_records(selection_records, flat_output)
        p1_output_hash = p1_outputs.hexdigest()
        p1_proposal_hash = p1_proposals.hexdigest()
        p3_proposal_hash = p3_proposals.hexdigest()
        p1_stream_preserved = p1_output_hash == P1_SELECTION_OUTPUT_STREAM_SHA256
        proposal_preserved = bool(
            torch_proposal_mismatches == 0
            and onnx_proposal_mismatches == 0
            and threshold_acceptance_mismatches == 0
            and p1_proposal_hash == p3_proposal_hash
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
            "p3_proposal_output_tensor_stream_sha256": p3_proposal_hash,
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
        p1_parity_passed = (
            p1_reexecution_parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        )
        frozen_preserved = frozen_before == frozen_after
        passed = bool(
            robust is not None
            and parity_passed
            and p1_parity_passed
            and frozen_preserved
            and proposal_preserved
            and training_role_mismatches == 0
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-relational-neighborhood-candidate-report.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": False,
            "objective": config["objective"],
            "training_evidence": training_evidence,
            "trainable_parameter_names": [],
            "frozen_p1_parameter_stream_sha256_before": frozen_before,
            "frozen_p1_parameter_stream_sha256_after": frozen_after,
            "frozen_p1_parameters_preserved": frozen_preserved,
            "training_role_matches": training_role_matches,
            "training_role_mismatches": training_role_mismatches,
            "p1_checkpoint_sha256": P1_CHECKPOINT_SHA256,
            "p1_onnx_sha256": P1_ONNX_SHA256,
            "p1_result_sha256": P1_RESULT_SHA256,
            "p1_report_sha256": P1_REPORT_SHA256,
            "p2_result_sha256": P2_RESULT_SHA256,
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
            "p3_preregistration_sha256": config["p3_preregistration_sha256"],
            "training_geometry_evidence_sha256": config[
                "training_geometry_evidence_sha256"
            ],
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
    "_p2_trigger_is_terminal",
    "preflight",
    "train_candidate",
]
