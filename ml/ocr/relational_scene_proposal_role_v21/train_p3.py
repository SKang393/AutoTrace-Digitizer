# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-shot zero-training selection runner for OCR V21 P3."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
import time
import tracemalloc
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import sha256_file
from .dataset import encode_scene, load_archive
from .model import RelationalSceneProposalRoleNet
from .train_p1 import (
    _choose_threshold,
    _evaluate_threshold,
    _gate_passed,
    _is_commit_ancestor,
    _read_json,
    _repository_head,
    _write_canonical,
    source_bundle_sha256,
)
from .protocol import THRESHOLDS


REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATE_ROOT = Path("ml/ocr/relational_scene_proposal_role_v21")
CONFIG_PATH = CANDIDATE_ROOT / "P3_CONFIG.json"
AUTHORIZATION_PATH = CANDIDATE_ROOT / "P3_SELECTION_AUTHORIZATION.json"
SEAL_PATH = CANDIDATE_ROOT / "SPLIT_SEAL.json"
P2_RESULT_PATH = CANDIDATE_ROOT / "P2_SELECTION_RESULT.json"
RESULT_PATH = CANDIDATE_ROOT / "P3_SELECTION_RESULT.json"
ATTEMPT_PATH = Path("artifacts/production-validation/ocr-v21-p3-attempt.json")
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v21-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v21-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v21-public.zip")
P2_CHECKPOINT_PATH = Path("artifacts/production-validation/ocr-v21-p2-checkpoint.pt")
P2_REPORT_PATH = Path("artifacts/production-validation/ocr-v21-p2-selection-report.json")
RUNNER_SOURCE_PATHS = (
    CONFIG_PATH,
    CANDIDATE_ROOT / "dataset.py",
    CANDIDATE_ROOT / "model.py",
    CANDIDATE_ROOT / "protocol.py",
    CANDIDATE_ROOT / "train_p1.py",
    CANDIDATE_ROOT / "train_p3.py",
)


class ScaledRelationalSceneProposalRoleNet(nn.Module):
    """Retain the exact base network and apply the fixed P3 output scale."""

    def __init__(self, base: RelationalSceneProposalRoleNet, scale: float) -> None:
        super().__init__()
        if scale <= 0.0 or scale >= 1.0:
            raise ValueError("OCR V21 P3 output scale must be between zero and one")
        self.base = base
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) * self.scale


def source_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in RUNNER_SOURCE_PATHS}


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _role_supported_score(proposal_logits: np.ndarray, role_logits: np.ndarray) -> np.ndarray:
    if proposal_logits.ndim != 2 or proposal_logits.shape[1] != 2:
        raise ValueError("OCR V21 P3 proposal logits must be [proposal_count,2]")
    if role_logits.ndim != 2 or role_logits.shape[0] != proposal_logits.shape[0]:
        raise ValueError("OCR V21 P3 role logits must share the proposal axis")
    proposal_probability = _softmax(proposal_logits)[:, 1]
    maximum_role_probability = _softmax(role_logits).max(axis=1)
    return np.sqrt(proposal_probability * maximum_role_probability)


def preflight() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected_config = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-calibration-candidate.v1",
        "revision": "graph-text-relational-scene-proposal-role-v21",
        "candidate_id": "P3",
        "candidate_type": "zero-training-role-supported-calibration",
        "expected_optimizer_steps": 0,
        "optimizer_steps_authorized": 0,
        "output_logit_scale": 0.5,
        "proposal_score": "sqrt(proposal_positive_probability * maximum_role_probability)",
        "selection_evaluation_limit": 1,
        "training_authorized": False,
        "selection_authorized": False,
        "public_execution_authorized": False,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V21 P3 candidate config field mismatch: {key}")
    if config.get("thresholds") != list(THRESHOLDS):
        raise RuntimeError("OCR V21 P3 candidate thresholds changed")

    authorization_path = REPO_ROOT / AUTHORIZATION_PATH
    if not authorization_path.is_file():
        raise RuntimeError("OCR V21 P3 selection is not authorized")
    authorization = _read_json(authorization_path)
    expected_authorization = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-selection-authorization.v1",
        "candidate_id": "P3",
        "execution_limit": 1,
        "execution_count": 0,
        "optimizer_steps_authorized": 0,
        "training_authorized": False,
        "selection_authorized": True,
        "public_execution_authorized": False,
        "private_validation_authorized": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected_authorization.items():
        if authorization.get(key) != value:
            raise RuntimeError(f"OCR V21 P3 authorization field mismatch: {key}")

    seal_path = REPO_ROOT / SEAL_PATH
    if sha256_file(seal_path) != config["split_seal_sha256"]:
        raise RuntimeError("OCR V21 split seal changed before P3")
    if authorization.get("split_seal_sha256") != config["split_seal_sha256"]:
        raise RuntimeError("OCR V21 P3 authorization does not bind the split seal")
    config_sha256 = sha256_file(REPO_ROOT / CONFIG_PATH)
    if authorization.get("candidate_config_sha256") != config_sha256:
        raise RuntimeError("OCR V21 P3 authorization does not bind the candidate config")
    hashes = source_hashes()
    if authorization.get("runner_source_sha256") != hashes:
        raise RuntimeError("OCR V21 P3 authorized runner sources changed")
    bundle = source_bundle_sha256(hashes)
    if authorization.get("runner_source_bundle_sha256") != bundle:
        raise RuntimeError("OCR V21 P3 authorized runner source bundle changed")

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
            raise RuntimeError(f"OCR V21 split seal state mismatch before P3: {key}")
    sealed_sources = seal.get("source_sha256")
    if not isinstance(sealed_sources, dict) or not sealed_sources:
        raise RuntimeError("OCR V21 split seal has no source hash inventory")
    for relative_path, expected_hash in sorted(sealed_sources.items()):
        source_path = REPO_ROOT / relative_path
        if not source_path.is_file() or sha256_file(source_path) != expected_hash:
            raise RuntimeError(f"OCR V21 sealed source changed before P3: {relative_path}")

    for key, path in (("train", TRAIN_ARCHIVE_PATH), ("selection", SELECTION_ARCHIVE_PATH)):
        expected_hash = seal[key]["archive_sha256"]
        if sha256_file(REPO_ROOT / path) != expected_hash or config[f"{key}_archive_sha256"] != expected_hash:
            raise RuntimeError(f"OCR V21 {key} archive changed before P3")
        if authorization.get(f"{key}_archive_sha256") != expected_hash:
            raise RuntimeError(f"OCR V21 P3 authorization does not bind the {key} archive")
    public_archive_sha256 = seal["public"]["archive_sha256"]
    if (
        sha256_file(REPO_ROOT / PUBLIC_ARCHIVE_PATH) != public_archive_sha256
        or config["public_archive_sha256"] != public_archive_sha256
        or authorization.get("public_archive_sha256") != public_archive_sha256
    ):
        raise RuntimeError("OCR V21 public archive changed before P3")

    prerequisite_paths = (
        (P2_CHECKPOINT_PATH, "p2_checkpoint_sha256"),
        (P2_REPORT_PATH, "p2_report_sha256"),
        (P2_RESULT_PATH, "p2_selection_result_sha256"),
    )
    for path, key in prerequisite_paths:
        expected_hash = config[key]
        if sha256_file(REPO_ROOT / path) != expected_hash or authorization.get(key) != expected_hash:
            raise RuntimeError(f"OCR V21 P3 prerequisite changed: {path.as_posix()}")
    p2_result = _read_json(REPO_ROOT / P2_RESULT_PATH)
    if (
        p2_result.get("p2_consumed") is not True
        or p2_result.get("selection_gate_passed") is not False
        or p2_result.get("public_archive_opened") is not False
        or p2_result.get("public_evaluation_count") != 0
    ):
        raise RuntimeError("OCR V21 P3 does not have the exact failed-closed P2 prerequisite")

    output_paths = (
        ATTEMPT_PATH,
        Path(config["onnx_output"]),
        Path(config["report_output"]),
        RESULT_PATH,
    )
    existing = [path.as_posix() for path in output_paths if (REPO_ROOT / path).exists()]
    if existing:
        raise RuntimeError("OCR V21 P3 is already executed or has output: " + ", ".join(existing))
    repository_head = _repository_head()
    authorized_source_commit = authorization.get("authorized_source_commit")
    if not isinstance(authorized_source_commit, str) or not _is_commit_ancestor(
        authorized_source_commit,
        repository_head,
    ):
        raise RuntimeError("OCR V21 P3 authorization does not bind an ancestor source commit")
    return {
        "authorization_sha256": sha256_file(authorization_path),
        "config": config,
        "config_sha256": config_sha256,
        "authorized_source_commit": authorized_source_commit,
        "repository_head": repository_head,
        "runner_source_sha256": hashes,
        "runner_source_bundle_sha256": bundle,
    }


def execute() -> dict[str, Any]:
    evidence = preflight()
    config = evidence["config"]
    attempt = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-attempt.v1",
        "candidate_id": "P3",
        "authorization_sha256": evidence["authorization_sha256"],
        "repository_head": evidence["repository_head"],
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_consumed": True,
        "optimizer_started": False,
        "optimizer_steps": 0,
        "completed": False,
        "production_approval": False,
        "release_eligible": False,
    }
    _write_canonical(REPO_ROOT / ATTEMPT_PATH, attempt)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = RelationalSceneProposalRoleNet(seed=2608152101)
    inherited = torch.load(REPO_ROOT / P2_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    if (
        inherited.get("candidate_id") != "P2"
        or inherited.get("optimizer_steps") != 1920
        or inherited.get("candidate_optimizer_steps") != 384
        or inherited.get("config_sha256") != "f166d9e7424e098b0c3e2770f061b3f2625f035f3aa14dd6ebc9aa1fbcc2c740"
        or inherited.get("authorization_sha256") != "53b14acb093db17844495fd37fa8e9ab7568a5b095744177fbd3f0b51ebdbd77"
    ):
        raise RuntimeError("OCR V21 P3 inherited checkpoint metadata is invalid")
    base.load_state_dict(inherited["model"], strict=True)
    model = ScaledRelationalSceneProposalRoleNet(base, float(config["output_logit_scale"])).eval()
    selection_scenes = load_archive(REPO_ROOT / SELECTION_ARCHIVE_PATH)
    sample, _candidates, _proposal, _roles = encode_scene(selection_scenes[0])
    onnx_path = REPO_ROOT / config["onnx_output"]
    start = time.perf_counter()
    tracemalloc.start()
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
    scores: list[np.ndarray] = []
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
            scores.append(_role_supported_score(ort_output[0, :, :2], ort_output[0, :, 2:]))
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
            tuple(scores),
            tuple(predicted_roles),
            tuple(proposal_labels),
            tuple(role_labels),
            threshold,
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
        "candidate_id": "P3",
        "execution_consumed": True,
        "execution_count": 1,
        "authorized_source_commit": evidence["authorized_source_commit"],
        "repository_head": evidence["repository_head"],
        "authorization_sha256": evidence["authorization_sha256"],
        "config_sha256": evidence["config_sha256"],
        "split_seal_sha256": config["split_seal_sha256"],
        "train_archive_sha256": config["train_archive_sha256"],
        "selection_archive_sha256": config["selection_archive_sha256"],
        "inherited_checkpoint_sha256": config["p2_checkpoint_sha256"],
        "optimizer_steps": 0,
        "output_logit_scale": config["output_logit_scale"],
        "proposal_score": config["proposal_score"],
        "onnx_sha256": sha256_file(onnx_path),
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
    "ScaledRelationalSceneProposalRoleNet",
    "_role_supported_score",
    "preflight",
    "source_hashes",
]
