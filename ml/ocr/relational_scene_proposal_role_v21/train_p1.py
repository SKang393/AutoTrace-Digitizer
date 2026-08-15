# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-shot CPU training and visible-selection runner for OCR V21 P1."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import encode_scene, load_archive
from .model import RelationalSceneProposalRoleNet
from .protocol import ROLE_ORDER, SEED, THRESHOLDS


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ROOT = Path("ml/ocr/relational_scene_proposal_role_v21")
CONFIG_PATH = CANDIDATE_ROOT / "P1_CONFIG.json"
AUTHORIZATION_PATH = CANDIDATE_ROOT / "P1_TRAINING_AUTHORIZATION.json"
SEAL_PATH = CANDIDATE_ROOT / "SPLIT_SEAL.json"
RESULT_PATH = CANDIDATE_ROOT / "P1_SELECTION_RESULT.json"
ATTEMPT_PATH = Path("artifacts/production-validation/ocr-v21-p1-attempt.json")
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v21-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v21-selection.zip")
RUNNER_SOURCE_PATHS = (
    CONFIG_PATH,
    CANDIDATE_ROOT / "dataset.py",
    CANDIDATE_ROOT / "model.py",
    CANDIDATE_ROOT / "protocol.py",
    CANDIDATE_ROOT / "train_p1.py",
)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    scene_count: int
    exact_scene_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    duplicate_regions: int
    prohibited_structure_hits: int
    role_accuracy: float
    per_role_accuracy: dict[str, float]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"OCR V21 expected a JSON object: {path}")
    return value


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in RUNNER_SOURCE_PATHS}


def source_bundle_sha256(hashes: dict[str, str]) -> str:
    digest = sha256()
    for path, value in sorted(hashes.items()):
        digest.update(f"{path}\0{value}\n".encode())
    return digest.hexdigest()


def _repository_head() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise RuntimeError("OCR V21 could not resolve the exact repository commit")
    return value


def _is_commit_ancestor(commit: str, head: str) -> bool:
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        return False
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, head),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected_config = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-candidate.v1",
        "revision": "graph-text-relational-scene-proposal-role-v21",
        "candidate_id": "P1",
        "epochs": 4,
        "expected_optimizer_steps": 1536,
        "selection_evaluation_limit": 1,
        "training_authorized": False,
        "public_execution_authorized": False,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V21 candidate config field mismatch: {key}")
    if config.get("thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V21 candidate thresholds changed")
    seal_path = REPO_ROOT / SEAL_PATH
    authorization_path = REPO_ROOT / AUTHORIZATION_PATH
    if not authorization_path.is_file():
        raise RuntimeError("OCR V21 P1 training is not authorized")
    authorization = _read_json(authorization_path)
    expected = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-training-authorization.v1",
        "candidate_id": "P1",
        "execution_limit": 1,
        "execution_count": 0,
        "training_authorized": True,
        "public_execution_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if authorization.get(key) != value:
            raise RuntimeError(f"OCR V21 authorization field mismatch: {key}")
    if sha256_file(seal_path) != config["split_seal_sha256"]:
        raise RuntimeError("OCR V21 split seal changed before P1")
    if authorization.get("split_seal_sha256") != config["split_seal_sha256"]:
        raise RuntimeError("OCR V21 authorization does not bind the split seal")
    if sha256_file(REPO_ROOT / CONFIG_PATH) != authorization.get("candidate_config_sha256"):
        raise RuntimeError("OCR V21 authorization does not bind the candidate config")
    hashes = source_hashes()
    if hashes != authorization.get("runner_source_sha256"):
        raise RuntimeError("OCR V21 authorized runner sources changed")
    bundle = source_bundle_sha256(hashes)
    if bundle != authorization.get("runner_source_bundle_sha256"):
        raise RuntimeError("OCR V21 authorized runner source bundle changed")
    seal = _read_json(seal_path)
    expected_seal = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-split-seal.v1",
        "candidate_id": "P1",
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected_seal.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V21 split seal state mismatch: {key}")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V21 split seal has no source hash inventory")
    for relative_path, expected_hash in sorted(sealed_sources.items()):
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise RuntimeError(f"OCR V21 sealed source changed before P1: {relative_path}")
    for key, path in (("train", TRAIN_ARCHIVE_PATH), ("selection", SELECTION_ARCHIVE_PATH)):
        expected_hash = seal[key]["archive_sha256"]
        if sha256_file(REPO_ROOT / path) != expected_hash or config[f"{key}_archive_sha256"] != expected_hash:
            raise RuntimeError(f"OCR V21 {key} archive changed before P1")
        if authorization.get(f"{key}_archive_sha256") != expected_hash:
            raise RuntimeError(f"OCR V21 authorization does not bind the {key} archive")
    if int(config["expected_optimizer_steps"]) > 1536:
        raise RuntimeError("OCR V21 optimizer budget exceeds the preregistered maximum")
    output_paths = (
        ATTEMPT_PATH,
        Path(config["checkpoint_output"]),
        Path(config["onnx_output"]),
        Path(config["report_output"]),
        RESULT_PATH,
    )
    existing = [path.as_posix() for path in output_paths if (REPO_ROOT / path).exists()]
    if existing:
        raise RuntimeError("OCR V21 P1 is already executed or has output: " + ", ".join(existing))
    repository_head = _repository_head()
    authorized_source_commit = authorization.get("authorized_source_commit")
    if not isinstance(authorized_source_commit, str) or not _is_commit_ancestor(
        authorized_source_commit, repository_head,
    ):
        raise RuntimeError("OCR V21 authorization does not bind an ancestor source commit")
    return {
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_path),
        "config": config,
        "config_sha256": sha256_file(REPO_ROOT / CONFIG_PATH),
        "authorized_source_commit": authorized_source_commit,
        "repository_head": repository_head,
        "runner_source_sha256": hashes,
        "runner_source_bundle_sha256": bundle,
        "seal": seal,
    }


def _balanced_proposal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = int((labels == 1).sum())
    negative = int((labels == 0).sum())
    if positive == 0 or negative == 0:
        raise RuntimeError("OCR V21 scene lacks a balanced proposal target")
    weights = torch.tensor((1.0 / negative, 1.0 / positive), dtype=torch.float32)
    weights *= 2.0 / weights.sum()
    return nn.functional.cross_entropy(logits, labels, weight=weights)


def _evaluate_threshold(
    probabilities: tuple[np.ndarray, ...],
    role_predictions: tuple[np.ndarray, ...],
    proposal_labels: tuple[np.ndarray, ...],
    role_labels: tuple[np.ndarray, ...],
    threshold: float,
) -> ThresholdMetrics:
    exact = true_positive = false_positive = false_negative = 0
    role_correct = {role: 0 for role in ROLE_ORDER}
    role_total = {role: 0 for role in ROLE_ORDER}
    for scores, predicted_roles, proposal_truth, role_truth in zip(
        probabilities, role_predictions, proposal_labels, role_labels, strict=True,
    ):
        accepted = scores >= threshold
        positives = proposal_truth == 1
        scene_tp = int(np.logical_and(accepted, positives).sum())
        scene_fp = int(np.logical_and(accepted, np.logical_not(positives)).sum())
        scene_fn = int(np.logical_and(np.logical_not(accepted), positives).sum())
        true_positive += scene_tp
        false_positive += scene_fp
        false_negative += scene_fn
        if scene_fp == 0 and scene_fn == 0:
            exact += 1
        for role_index, role in enumerate(ROLE_ORDER):
            truth_mask = role_truth == role_index
            role_total[role] += int(truth_mask.sum())
            role_correct[role] += int(np.logical_and.reduce((truth_mask, accepted, predicted_roles == role_index)).sum())
    per_role = {role: role_correct[role] / max(1, role_total[role]) for role in ROLE_ORDER}
    total_roles = sum(role_total.values())
    return ThresholdMetrics(
        threshold,
        len(probabilities),
        exact,
        true_positive,
        false_positive,
        false_negative,
        0,
        false_positive,
        sum(role_correct.values()) / max(1, total_roles),
        per_role,
    )


def _gate_passed(metrics: ThresholdMetrics) -> bool:
    return (
        metrics.exact_scene_count == metrics.scene_count
        and metrics.false_positives == 0
        and metrics.false_negatives == 0
        and metrics.duplicate_regions == 0
        and metrics.prohibited_structure_hits == 0
        and metrics.role_accuracy >= 0.90
        and min(metrics.per_role_accuracy.values()) >= 0.90
    )


def _choose_threshold(metrics: tuple[ThresholdMetrics, ...]) -> ThresholdMetrics:
    passing = [value for value in metrics if _gate_passed(value)]
    if passing:
        return passing[0]
    return max(
        metrics,
        key=lambda value: (
            value.exact_scene_count,
            -(value.false_positives + value.false_negatives),
            value.role_accuracy,
            -value.threshold,
        ),
    )


def execute() -> dict[str, Any]:
    evidence = preflight()
    config = evidence["config"]
    attempt = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-attempt.v1",
        "candidate_id": "P1",
        "authorization_sha256": evidence["authorization_sha256"],
        "repository_head": evidence["repository_head"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_consumed": True,
        "optimizer_started": False,
        "completed": False,
        "production_approval": False,
        "release_eligible": False,
    }
    _write_canonical(REPO_ROOT / ATTEMPT_PATH, attempt)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(config["seed"]))
    np.random.seed(int(config["seed"]) % (2**32))
    model = RelationalSceneProposalRoleNet(seed=int(config["seed"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    train_scenes = load_archive(REPO_ROOT / TRAIN_ARCHIVE_PATH)
    selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
    order_generator = torch.Generator().manual_seed(int(config["seed"]) + 91)
    optimizer_steps = 0
    epoch_losses: list[float] = []
    start = time.perf_counter()
    tracemalloc.start()
    attempt["optimizer_started"] = True
    _write_canonical(REPO_ROOT / ATTEMPT_PATH, attempt)
    model.train()
    for _epoch in range(int(config["epochs"])):
        total_loss = 0.0
        for scene_index in torch.randperm(len(train_scenes), generator=order_generator).tolist():
            encoded, _candidates, proposal_truth, role_truth = encode_scene(train_scenes[scene_index])
            value = torch.from_numpy(encoded).unsqueeze(0)
            proposal_target = torch.from_numpy(proposal_truth)
            role_target = torch.from_numpy(role_truth)
            optimizer.zero_grad(set_to_none=True)
            output = model(value)[0]
            proposal_loss = _balanced_proposal_loss(output[:, :2], proposal_target)
            positive = proposal_target == 1
            role_loss = nn.functional.cross_entropy(output[positive, 2:], role_target[positive])
            loss = proposal_loss + float(config["role_loss_weight"]) * role_loss
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total_loss += float(loss.detach())
        epoch_losses.append(total_loss / len(train_scenes))
        print(
            f"OCR V21 P1 epoch {_epoch + 1}/{config['epochs']} complete; "
            f"optimizer_steps={optimizer_steps}",
            flush=True,
        )
    if optimizer_steps != int(config["expected_optimizer_steps"]):
        raise RuntimeError(f"OCR V21 optimizer step mismatch: {optimizer_steps}")

    checkpoint_path = REPO_ROOT / config["checkpoint_output"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_steps": optimizer_steps,
        "candidate_id": "P1",
        "config_sha256": evidence["config_sha256"],
        "authorization_sha256": evidence["authorization_sha256"],
    }, checkpoint_path)

    model.eval()
    onnx_path = REPO_ROOT / config["onnx_output"]
    sample, _candidates, _proposal, _roles = encode_scene(selection_scenes[0])
    torch.onnx.export(
        model,
        torch.from_numpy(sample).unsqueeze(0),
        str(onnx_path),
        input_names=["proposals"],
        output_names=["logits"],
        dynamic_axes={"proposals": {1: "proposal_count"}, "logits": {1: "proposal_count"}},
        opset_version=17,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    probabilities: list[np.ndarray] = []
    predicted_roles: list[np.ndarray] = []
    proposal_labels: list[np.ndarray] = []
    role_labels: list[np.ndarray] = []
    runtime_calls: list[dict[str, Any]] = []
    maximum_parity = 0.0
    with torch.inference_mode():
        for scene in selection_scenes:
            encoded, _candidates, proposal_truth, role_truth = encode_scene(scene)
            value = encoded[None, ...]
            torch_output = model(torch.from_numpy(value)).numpy()
            ort_output = session.run(["logits"], {"proposals": value})[0]
            maximum_parity = max(maximum_parity, float(np.max(np.abs(torch_output - ort_output))))
            shifted = ort_output[0, :, :2] - ort_output[0, :, :2].max(axis=1, keepdims=True)
            proposal_probability = np.exp(shifted)[:, 1] / np.exp(shifted).sum(axis=1)
            probabilities.append(proposal_probability)
            predicted_roles.append(ort_output[0, :, 2:].argmax(axis=1))
            proposal_labels.append(proposal_truth)
            role_labels.append(role_truth)
            runtime_calls.append({
                "input_sha256": sha256(value.tobytes(order="C")).hexdigest(),
                "input_shape": list(value.shape),
                "output_sha256": sha256(ort_output.tobytes(order="C")).hexdigest(),
                "output_shape": list(ort_output.shape),
                "provider": "CPUExecutionProvider",
            })
    threshold_metrics = tuple(
        _evaluate_threshold(
            tuple(probabilities), tuple(predicted_roles), tuple(proposal_labels), tuple(role_labels), threshold,
        )
        for threshold in config["thresholds"]
    )
    selected = _choose_threshold(threshold_metrics)
    parity_passed = maximum_parity <= 0.00001
    selection_passed = _gate_passed(selected) and parity_passed
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed = time.perf_counter() - start
    report = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-selection-report.v1",
        "candidate_id": "P1",
        "execution_consumed": True,
        "execution_count": 1,
        "authorized_source_commit": evidence["authorized_source_commit"],
        "repository_head": evidence["repository_head"],
        "authorization_sha256": evidence["authorization_sha256"],
        "config_sha256": evidence["config_sha256"],
        "split_seal_sha256": config["split_seal_sha256"],
        "train_archive_sha256": config["train_archive_sha256"],
        "selection_archive_sha256": config["selection_archive_sha256"],
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_sha256": sha256_file(onnx_path),
        "optimizer_steps": optimizer_steps,
        "epoch_losses": epoch_losses,
        "selected_threshold": selected.threshold,
        "selected_metrics": asdict(selected),
        "threshold_metrics": [asdict(value) for value in threshold_metrics],
        "selection_gate_passed": selection_passed,
        "onnx_parity_max_abs": maximum_parity,
        "onnx_parity_passed": parity_passed,
        "provider": "CPUExecutionProvider",
        "runtime_calls": runtime_calls,
        "runtime_call_count": len(runtime_calls),
        "elapsed_seconds": elapsed,
        "python_peak_traced_bytes": peak_bytes,
        "python_current_traced_bytes": current_bytes,
        "toolchain": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
        },
        "runner_source_sha256": evidence["runner_source_sha256"],
        "runner_source_bundle_sha256": evidence["runner_source_bundle_sha256"],
        "public_archive_opened": False,
        "public_execution_count": 0,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "manifest_creation_authorized": False,
        "model_store_promotion_authorized": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    report_path = REPO_ROOT / config["report_output"]
    _write_canonical(report_path, report)
    attempt.update({
        "completed": True,
        "optimizer_steps": optimizer_steps,
        "selection_gate_passed": selection_passed,
        "report_sha256": sha256_file(report_path),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _write_canonical(REPO_ROOT / ATTEMPT_PATH, attempt)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight()
        print(json.dumps({
            "authorization_sha256": evidence["authorization_sha256"],
            "repository_head": evidence["repository_head"],
            "runner_source_bundle_sha256": evidence["runner_source_bundle_sha256"],
            "ready": True,
        }, sort_keys=True))
        return 0
    report = execute()
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    print(json.dumps({
        "checkpoint_sha256": report["checkpoint_sha256"],
        "onnx_sha256": report["onnx_sha256"],
        "optimizer_steps": report["optimizer_steps"],
        "report_sha256": sha256_file(REPO_ROOT / config["report_output"]),
        "selection_gate_passed": report["selection_gate_passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RUNNER_SOURCE_PATHS",
    "ThresholdMetrics",
    "_choose_threshold",
    "_evaluate_threshold",
    "_gate_passed",
    "preflight",
    "source_bundle_sha256",
    "source_hashes",
]
