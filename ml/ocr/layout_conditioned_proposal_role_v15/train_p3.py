# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-step anchor-preserving parity repair for OCR V15 P3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, encode_proposal, proposals, split_fingerprint
from .model import LayoutConditionedProposalRoleNet
from .model_p3 import AnchorScaledCandidate
from .pipeline import evaluate_thresholds
from .protocol import (
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    ROLE_CLASS_ACCURACY_MINIMUM,
    TASK,
)
from .train_p1 import _export


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/layout_conditioned_proposal_role_v15")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P2_REPORT_PATH = ROOT / "artifacts/P2-run/candidate-report.json"
P2_CHECKPOINT_PATH = ROOT / "artifacts/P2-run/graph-text-layout-conditioned-proposal-role-v15-p2.pt"
P2_ONNX_PATH = ROOT / "artifacts/P2-run/graph-text-layout-conditioned-proposal-role-v15-p2.onnx"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "model_p3.py", ROOT / "pipeline.py",
    ROOT / "protocol.py", ROOT / "train_p1.py", ROOT / "train_p3.py",
    Path("ml/ocr/structural_graph_proposal_role_v14/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model_p2.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _load_trigger(config: dict[str, object]) -> dict[str, object]:
    if sha256_file(REPO_ROOT / P1_RESULT_PATH) != config["p1_result_sha256"]:
        raise RuntimeError("OCR V15 P1 result changed before P3")
    if sha256_file(REPO_ROOT / P2_RESULT_PATH) != config["p2_result_sha256"]:
        raise RuntimeError("OCR V15 P2 result changed before P3")
    result = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
    if (
        result.get("status") != "failed_selection"
        or result.get("consumed") is not True
        or result.get("public_gate_archive_opened") is not False
        or result.get("public_gate_evaluations") != 0
        or result.get("production_approval") is not False
        or result.get("release_eligible") is not False
    ):
        raise RuntimeError("OCR V15 P2 is not exact consumed unopened evidence")
    if (
        result["candidate_report_sha256"] != config["p2_report_sha256"]
        or sha256_file(REPO_ROOT / P2_REPORT_PATH) != config["p2_report_sha256"]
        or result["checkpoint_sha256"] != config["p2_checkpoint_sha256"]
        or sha256_file(REPO_ROOT / P2_CHECKPOINT_PATH) != config["p2_checkpoint_sha256"]
        or result["onnx_sha256"] != config["p2_onnx_sha256"]
        or result["onnx_path"] != config["p2_onnx_path"]
        or sha256_file(REPO_ROOT / P2_ONNX_PATH) != config["p2_onnx_sha256"]
    ):
        raise RuntimeError("OCR V15 P2 payload binding changed before P3")
    trigger = config["p2_aggregate_trigger"]
    selected = result["selection_metrics"]
    for key in (
        "exact_scene_count", "true_positives", "false_positives", "false_negatives",
        "duplicate_region_count", "prohibited_structure_hits", "role_accuracy",
    ):
        if selected[key] != trigger[key]:
            raise RuntimeError(f"OCR V15 P2 aggregate trigger changed: {key}")
    if (
        result["selected_threshold"] != trigger["selected_threshold"]
        or result["onnx_parity_maximum_absolute_error"] != trigger["onnx_parity_maximum_absolute_error"]
        or result["role_invariance_maximum_absolute_error"] != 0.0
    ):
        raise RuntimeError("OCR V15 P2 calibration trigger changed")
    return result


def _probability(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent[:, 1] / exponent.sum(axis=1)


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V15 P3 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started, phase = time.perf_counter(), "initialization"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V15 P3 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V15 P3 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V15 P3 public seal changed")
        _load_trigger(config)
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        validation = build_split("validation")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V15 P3 validation split changed")

        base = LayoutConditionedProposalRoleNet(seed=int(config["base_seed"]))
        state = torch.load(REPO_ROOT / P2_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        base.load_state_dict(state["state_dict"], strict=True)
        base.eval()
        candidate = AnchorScaledCandidate(
            base, scale=float(config["output_scale"]), anchor=float(config["anchor_threshold"]),
        ).eval()
        for parameter in candidate.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in candidate.parameters()):
            raise RuntimeError("OCR V15 P3 must have no trainable parameters")

        phase = "equivalence"
        acceptance_mismatches = role_argmax_mismatches = proposal_count = 0
        example: torch.Tensor | None = None
        with torch.inference_mode():
            for scene in validation:
                scene_proposals = proposals(scene.raster)
                values = np.stack([
                    encode_proposal(scene.raster, proposal, scene.plot) for proposal in scene_proposals
                ]).astype(np.float32)
                tensor = torch.from_numpy(values)
                if example is None:
                    example = tensor
                base_output = base(tensor).numpy()
                candidate_output = candidate(tensor).numpy()
                base_accepted = _probability(base_output[:, :2]) >= float(config["anchor_threshold"])
                candidate_accepted = _probability(candidate_output[:, :2]) >= float(config["anchor_threshold"])
                acceptance_mismatches += int(np.count_nonzero(base_accepted != candidate_accepted))
                role_argmax_mismatches += int(np.count_nonzero(
                    np.argmax(base_output[:, 2:], axis=1) != np.argmax(candidate_output[:, 2:], axis=1)
                ))
                proposal_count += len(values)
        if example is None:
            raise RuntimeError("OCR V15 P3 validation produced no export example")

        phase = "export"
        onnx_path = output_dir / "graph-text-layout-conditioned-proposal-role-v15-p3.onnx"
        _export(candidate, example, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V15 P3 selection requires CPU execution only")
        parity_values = example[:min(256, len(example))]
        with torch.inference_mode():
            expected = candidate(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        calls = 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.asarray(
                session.run(None, {"region_proposals": np.ascontiguousarray(input_values)})[0], dtype=np.float32,
            )

        phase = "selection"
        comparisons = evaluate_thresholds(
            validation, runner, tuple(float(value) for value in config["selection_thresholds"]),
        )
        selected = max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["role_accuracy"], item["threshold"],
        ))
        metrics = selected["metrics"]
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["true_positives"] == metrics["truth_region_count"]
            and metrics["false_positives"] == metrics["false_negatives"]
            == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
            and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
            and min(metrics["per_role_accuracy"].values()) >= ROLE_CLASS_ACCURACY_MINIMUM
            and parity_passed and calls == len(validation)
            and acceptance_mismatches == role_argmax_mismatches == 0
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-layout-conditioned-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v14_validation_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p2_validation_case_detail_or_pixels_used_for_design": False,
            "p2_aggregate_metrics_only_used_for_design": True,
            "optimizer_steps": 0, "weights_changed": False,
            "isolated_change": config["isolated_change"],
            "p1_result_sha256": config["p1_result_sha256"],
            "p2_result_sha256": config["p2_result_sha256"],
            "p2_checkpoint_sha256": config["p2_checkpoint_sha256"],
            "output_scale": config["output_scale"], "anchor_threshold": config["anchor_threshold"],
            "anchor_equivalence_proposal_count": proposal_count,
            "anchor_acceptance_mismatch_count": acceptance_mismatches,
            "role_argmax_mismatch_count": role_argmax_mismatches,
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": maximum_error, "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider", "threshold_comparisons": comparisons,
            "selected_threshold": selected["threshold"], "selection_metrics": metrics,
            "direct_execution_inference_calls": calls,
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-layout-conditioned-proposal-role-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": 0,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False, "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False, "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = train_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"], "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"],
        "anchor_acceptance_mismatch_count": report["anchor_acceptance_mismatch_count"],
        "role_argmax_mismatch_count": report["role_argmax_mismatch_count"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
