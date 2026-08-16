# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use constrained continuation and visible selection for OCR V22 P3."""

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
from .model import SceneEvidenceAttentionNet
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
    TASK,
    THRESHOLDS,
)
from . import train_p1 as common
from . import train_p2 as p2


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/scene_evidence_attention_v22")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
PROTOCOL_PATH = ROOT / "P3_PROTOCOL.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_RESULT_SHA256 = "ab5942d4f4d42531897f95ca2fe949101d8c46ad7f9a54c8fcc66a6687ed8f46"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v22-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v22-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v22-public.zip")
RUNNER_SOURCE_PATHS = tuple(dict.fromkeys((
    PROTOCOL_PATH,
    ROOT / "train_p3.py",
    *p2.RUNNER_SOURCE_PATHS,
)))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"OCR V22 expected a JSON object: {path}")
    return value


def _set_trainable(model: SceneEvidenceAttentionNet, prefix: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefix))


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-scene-evidence-attention-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "recognition-conditioned-complete-proposal-set-attention-v1",
        "objective": "two_phase_proposal_continuation_then_frozen_role_head_repair_v1",
        "model_license": "Apache-2.0",
        "seed": 2_608_162_203,
        "proposal_continuation_epochs": 3,
        "role_head_repair_epochs": 2,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1280,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 100000,
        "detector_prefilter_applied": False,
        "detector_path": DETECTOR_PATH,
        "detector_sha256": DETECTOR_SHA256,
        "recognizer_path": RECOGNIZER_PATH,
        "recognizer_sha256": RECOGNIZER_SHA256,
        "recognizer_inference_yaml_path": RECOGNIZER_YAML_PATH,
        "recognizer_inference_yaml_sha256": RECOGNIZER_YAML_SHA256,
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_result_sha256": P2_RESULT_SHA256,
        "validation_or_public_pixels_used_for_training": False,
        "selection_evaluation_limit": 1,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V22 P3 config field mismatch: {key}")
    if config.get("selection_thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V22 P3 thresholds changed")
    if (
        int(config["proposal_continuation_epochs"])
        + int(config["role_head_repair_epochs"])
    ) * 256 != int(config["expected_optimizer_steps"]):
        raise RuntimeError("OCR V22 P3 optimizer schedule changed")
    if int(config["expected_optimizer_steps"]) > 1280:
        raise RuntimeError("OCR V22 P3 optimizer budget exceeds the preregistration")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V22 P3 runner source bundle changed")
    if sha256_file(REPO_ROOT / PROTOCOL_PATH) != config.get("protocol_sha256"):
        raise RuntimeError("OCR V22 P3 protocol changed")
    if sha256_file(REPO_ROOT / P2_RESULT_PATH) != P2_RESULT_SHA256:
        raise RuntimeError("OCR V22 P2 aggregate result changed before P3")
    p2_result = _read_json(REPO_ROOT / P2_RESULT_PATH)
    p2_expected = {
        "candidate_id": "P2",
        "candidate_consumed": True,
        "status": "failed_selection",
        "selection_gate_passed": False,
        "public_gate_archive_opened": False,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in p2_expected.items():
        if p2_result.get(key) != value:
            raise RuntimeError(f"OCR V22 P2 result is not a fail-closed P3 trigger: {key}")
    exact_p2 = {
        str(config["p2_checkpoint_path"]): str(config["p2_checkpoint_sha256"]),
        str(config["p2_onnx_path"]): str(config["p2_onnx_sha256"]),
    }
    if exact_p2[str(config["p2_checkpoint_path"])] != p2_result.get("checkpoint_sha256"):
        raise RuntimeError("OCR V22 P3 checkpoint binding differs from the P2 result")
    if exact_p2[str(config["p2_onnx_path"])] != p2_result.get("onnx_sha256"):
        raise RuntimeError("OCR V22 P3 ONNX binding differs from the P2 result")
    for relative, expected_hash in exact_p2.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V22 exact P2 payload changed before P3: {relative}")
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V22 split seal changed before P3")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    seal_expected = {
        "schema": "graphreader.ocr-scene-evidence-attention-split-seal.v1",
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
            raise RuntimeError(f"OCR V22 split seal field changed: {key}")
    head = common._repository_head()
    if not common._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V22 split seal source commit is not an ancestor")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V22 split seal source inventory is absent")
    for relative, expected_hash in sealed_sources.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V22 frozen split source changed: {relative}")
    archive_bindings = {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }
    for split, (path, config_key) in archive_bindings.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(config_key):
            raise RuntimeError(f"OCR V22 {split} archive changed before P3")
    exact_inputs = {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
    }
    for relative, expected_hash in exact_inputs.items():
        if sha256_file(REPO_ROOT / relative) != expected_hash:
            raise RuntimeError(f"OCR V22 frozen model input changed: {relative}")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V22 P3 output already exists")
    return {"config": config, "head": head, "seal": seal, "p2_result": p2_result}


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V22 P3 output exists: {output_dir}")
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
            raise RuntimeError("OCR V22 train stored fixtures violate the seal")

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
            raise RuntimeError("OCR V22 P3 training feature width changed")
        train_groups = common._feature_groups(train_records, len(train_scenes))
        train_roles = common._role_targets(train_scenes, train_records)
        registered_training = evidence["seal"]["splits"]["train"]["proposal_summary"]
        for key in ("proposal_count", "positive_proposal_count", "negative_proposal_count"):
            if training_evidence[key] != registered_training[key]:
                raise RuntimeError(f"OCR V22 P3 incomplete training proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in train_scenes)
        if int(train_labels.sum()) != expected_truths or np.any(train_roles[train_labels == 1] < 0):
            raise RuntimeError("OCR V22 P3 production stream omitted a training truth")

        phase = "load_exact_p2_checkpoint"
        model = SceneEvidenceAttentionNet(seed=2_608_162_202)
        checkpoint = torch.load(
            REPO_ROOT / str(config["p2_checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise RuntimeError("OCR V22 P2 checkpoint does not contain the exact state dictionary")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        generator = common._configure(int(config["seed"]))
        positives = int(train_labels.sum())
        negatives = int(len(train_labels) - positives)
        if positives == 0 or negatives == 0:
            raise RuntimeError("OCR V22 P3 requires positive and negative production proposals")
        proposal_weights = torch.tensor((1.0 / negatives, 1.0 / positives), dtype=torch.float32)
        proposal_weights *= 2.0 / proposal_weights.sum()
        losses: list[dict[str, float | int | str]] = []

        phase = "proposal_only_continuation"
        _set_trainable(model, "")
        for parameter in model.role_head.parameters():
            parameter.requires_grad_(False)
        proposal_optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["proposal_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        model.train()
        for epoch in range(int(config["proposal_continuation_epochs"])):
            epoch_loss = 0.0
            epoch_negative_margin = 0.0
            epoch_positive_margin = 0.0
            for scene_index in torch.randperm(len(train_groups), generator=generator).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                proposal_targets = torch.from_numpy(train_labels[indices])
                output = model(values)[0]
                loss, negative_margin, positive_margin = p2._proposal_objective(
                    output[:, :2], proposal_targets, proposal_weights, config,
                )
                proposal_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                proposal_optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
                epoch_negative_margin += float(negative_margin.detach())
                epoch_positive_margin += float(positive_margin.detach())
            count = len(train_groups)
            losses.append({
                "phase": "proposal_only_continuation",
                "epoch": epoch + 1,
                "loss": epoch_loss / count,
                "negative_scene_extrema_margin": epoch_negative_margin / count,
                "positive_scene_extrema_margin": epoch_positive_margin / count,
            })
            print(
                f"OCR V22 P3 proposal epoch {epoch + 1}/{config['proposal_continuation_epochs']} "
                f"complete; optimizer_steps={optimizer_steps}",
                flush=True,
            )

        phase = "frozen_role_head_repair"
        _set_trainable(model, "role_head.")
        role_optimizer = torch.optim.AdamW(
            model.role_head.parameters(),
            lr=float(config["role_learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        for epoch in range(int(config["role_head_repair_epochs"])):
            epoch_loss = 0.0
            for scene_index in torch.randperm(len(train_groups), generator=generator).tolist():
                indices = train_groups[scene_index]
                values = torch.from_numpy(train_values[indices]).unsqueeze(0)
                proposal_targets = torch.from_numpy(train_labels[indices])
                role_targets = torch.from_numpy(train_roles[indices])
                positive = proposal_targets == 1
                output = model(values)[0]
                loss = nn.functional.cross_entropy(output[positive, 2:], role_targets[positive])
                role_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                role_optimizer.step()
                optimizer_steps += 1
                epoch_loss += float(loss.detach())
            losses.append({
                "phase": "frozen_role_head_repair",
                "epoch": epoch + 1,
                "loss": epoch_loss / len(train_groups),
            })
            print(
                f"OCR V22 P3 role epoch {epoch + 1}/{config['role_head_repair_epochs']} "
                f"complete; optimizer_steps={optimizer_steps}",
                flush=True,
            )
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V22 P3 optimizer-step count changed")

        phase = "checkpoint_and_onnx_export"
        checkpoint_path = output_dir / "graph-text-scene-evidence-attention-v22-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-scene-evidence-attention-v22-p3.onnx"
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
            raise RuntimeError("OCR V22 validation stored fixtures violate the seal")
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
                raise RuntimeError(f"OCR V22 P3 incomplete selection proposal stream: {key}")
        expected_truths = sum(len(scene.truths) for scene in selection_scenes)
        if sum(record.truth_index >= 0 for record in selection_records) != expected_truths:
            raise RuntimeError("OCR V22 P3 production stream omitted a validation truth")
        onnx_outputs: list[np.ndarray] = []
        candidate_inputs = sha256()
        candidate_outputs = sha256()
        parity_error = 0.0
        with torch.inference_mode():
            for indices in selection_groups:
                values = np.ascontiguousarray(selection_values[indices][None, ...])
                expected_output = model(torch.from_numpy(values)).numpy()
                actual_output = np.asarray(
                    candidate_session.run(None, {candidate_input: values})[0], dtype=np.float32,
                )
                parity_error = max(
                    parity_error, float(np.max(np.abs(expected_output - actual_output))),
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
            "schema": "graphreader.ocr-scene-evidence-attention-candidate-report.v1",
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
                    "proposal_continuation_epochs",
                    "proposal_learning_rate",
                    "role_head_repair_epochs",
                    "role_learning_rate",
                )
            },
            "p2_result_sha256": P2_RESULT_SHA256,
            "p2_checkpoint_sha256": config["p2_checkpoint_sha256"],
            "p2_onnx_sha256": config["p2_onnx_sha256"],
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
            "schema": "graphreader.ocr-scene-evidence-attention-failure.v1",
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
    "_set_trainable",
    "preflight",
    "train_candidate",
]
