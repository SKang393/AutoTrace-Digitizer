# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P2 calibration for OCR V16."""

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
from .model import MarginRobustLayoutProposalRoleNet
from .model_p2 import CalibratedMarginCandidate
from .pipeline import evaluate_thresholds
from .protocol import ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION, TASK
from .train_p1 import _export, _select_robust_window


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/margin_robust_layout_proposal_role_v16")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "P1_RESULT.json", ROOT / "dataset.py", ROOT / "model.py", ROOT / "model_p2.py",
    ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "train_p1.py", ROOT / "train_p2.py",
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model_p2.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _parity_values(validation: tuple[object, ...], count: int = 256) -> np.ndarray:
    values: list[np.ndarray] = []
    for scene in validation:
        for candidate in proposals(scene.raster):
            values.append(encode_proposal(scene.raster, candidate, scene.plot))
            if len(values) == count:
                return np.stack(values).astype(np.float32)
    raise RuntimeError("OCR V16 P2 validation split did not provide enough parity proposals")


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V16 P2 output exists: {output_dir}")
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
            raise RuntimeError("OCR V16 P2 runner sources changed")
        p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / P1_RESULT_PATH) != config["p1_result_sha256"]:
            raise RuntimeError("OCR V16 P1 result changed")
        p1_report_path = REPO_ROOT / p1["candidate_report_path"]
        checkpoint_path = REPO_ROOT / p1["checkpoint_path"]
        if sha256_file(p1_report_path) != p1["candidate_report_sha256"]:
            raise RuntimeError("OCR V16 P1 candidate report changed")
        if sha256_file(checkpoint_path) != p1["checkpoint_sha256"]:
            raise RuntimeError("OCR V16 P1 checkpoint changed")
        if p1["status"] != "failed_selection" or p1["consumed"] is not True:
            raise RuntimeError("OCR V16 P2 requires the exact consumed P1 selection result")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        validation = build_split("validation")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V16 P2 validation split changed")

        base = MarginRobustLayoutProposalRoleNet(seed=int(config["seed"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base.load_state_dict(checkpoint["state_dict"], strict=True)
        base.eval()
        model = CalibratedMarginCandidate(
            base, positive_logit_bias=float(config["positive_logit_bias"]),
        ).eval()
        parity_numpy = _parity_values(validation)
        parity_values = torch.from_numpy(parity_numpy)
        with torch.inference_mode():
            base_output = base(parity_values)
            expected = model(parity_values).numpy()
        if float(torch.max(torch.abs(base_output[:, 2:] - torch.from_numpy(expected)[:, 2:]))) != 0.0:
            raise RuntimeError("OCR V16 P2 changed role logits")
        phase = "export"
        onnx_path = output_dir / "graph-text-margin-robust-layout-proposal-role-v16-p2.onnx"
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V16 P2 selection requires CPU execution only")
        actual = np.asarray(session.run(None, {"region_proposals": parity_numpy})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        role_argmax_mismatches = int(np.count_nonzero(
            np.argmax(base_output[:, 2:].numpy(), axis=1) != np.argmax(actual[:, 2:], axis=1)
        ))
        calls = 0

        def runner(values: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.asarray(
                session.run(None, {"region_proposals": np.ascontiguousarray(values)})[0], dtype=np.float32,
            )

        phase = "selection"
        comparisons = evaluate_thresholds(
            validation, runner, tuple(float(value) for value in config["selection_thresholds"]),
        )
        robust = _select_robust_window(comparisons)
        selected = robust[0] if robust else max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["role_accuracy"], item["threshold"],
        ))
        window = robust[1] if robust else ()
        passed = robust is not None and parity_passed and role_argmax_mismatches == 0 and calls == len(validation)
        report: dict[str, object] = {
            "schema": "graphreader.ocr-margin-robust-layout-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v15_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_aggregate_metrics_only_used_for_design": True,
            "p1_validation_case_detail_or_pixels_used_for_design": False,
            "p1_checkpoint_path": p1["checkpoint_path"], "p1_checkpoint_sha256": p1["checkpoint_sha256"],
            "optimizer_steps": 0, "weights_changed": False,
            "positive_logit_bias": config["positive_logit_bias"],
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": maximum_error, "onnx_parity_passed": parity_passed,
            "role_argmax_mismatch_count": role_argmax_mismatches,
            "provider": "CPUExecutionProvider", "threshold_comparisons": comparisons,
            "minimum_consecutive_passing_thresholds": config["minimum_consecutive_passing_thresholds"],
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"], "selection_metrics": selected["metrics"],
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
            "schema": "graphreader.ocr-margin-robust-layout-proposal-role-failure.v1",
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
    report = evaluate_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"], "optimizer_steps": report["optimizer_steps"],
        "positive_logit_bias": report["positive_logit_bias"],
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
