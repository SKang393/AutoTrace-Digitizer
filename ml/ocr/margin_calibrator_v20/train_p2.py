# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use floor-free P2 selection for the exact consumed V20 P1 model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_split_archive, proposal_summary, split_fingerprint
from .model import MarginSeparatedProposalCalibrator
from .pipeline import evaluate_thresholds, extract_features, select_robust_window
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/margin_calibrator_v20")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "model.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "train_p2.py",
    P1_RESULT_PATH,
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V20 P2 requires CPUExecutionProvider only")
    return session


def evaluate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V20 P2 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
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
    phase = "preflight"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(
            REPO_ROOT, RUNNER_SOURCE_PATHS
        ):
            raise RuntimeError("OCR V20 P2 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V20 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V20 public seal changed")
        if (
            config["trigger_result_path"] != P1_RESULT_PATH.as_posix()
            or config["trigger_result_sha256"] != sha256_file(REPO_ROOT / P1_RESULT_PATH)
        ):
            raise RuntimeError("OCR V20 aggregate P1 result changed")
        p1_result = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
        if (
            p1_result["status"] != "failed_selection"
            or p1_result["case_level_details_emitted"] is not False
            or p1_result["public_gate_archive_opened"] is not False
            or p1_result["public_gate_evaluations"] != 0
            or config["base_checkpoint_path"] != p1_result["checkpoint_path"]
            or config["base_checkpoint_sha256"] != p1_result["checkpoint_sha256"]
            or config["base_onnx_path"] != p1_result["onnx_path"]
            or config["base_onnx_sha256"] != p1_result["onnx_sha256"]
            or config["expected_optimizer_steps"] != 0
            or config["weights_changed"] is not False
        ):
            raise RuntimeError("OCR V20 P1 trigger is not the consumed aggregate-only failure")
        exact_inputs = {
            DETECTOR_PATH: DETECTOR_SHA256,
            RECOGNIZER_PATH: RECOGNIZER_SHA256,
            RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
            config["base_checkpoint_path"]: config["base_checkpoint_sha256"],
            config["base_onnx_path"]: config["base_onnx_sha256"],
        }
        for relative, expected in exact_inputs.items():
            if sha256_file(REPO_ROOT / relative) != expected:
                raise RuntimeError(f"OCR V20 P2 exact input changed: {relative}")

        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        registered = selection["validation"]
        archive = REPO_ROOT / registered["fixture_archive_path"]
        manifest = REPO_ROOT / registered["private_manifest_path"]
        if (
            sha256_file(archive) != registered["fixture_archive_sha256"]
            or sha256_file(manifest) != registered["private_manifest_sha256"]
        ):
            raise RuntimeError("OCR V20 stored validation bytes changed")
        scenes = load_split_archive(archive, manifest, expected_split="validation")
        summary = proposal_summary(scenes)
        if (
            split_fingerprint(scenes) != registered["split_fingerprint"]
            or any(summary[key] != registered[key] for key in summary)
        ):
            raise RuntimeError("OCR V20 stored validation fixtures violate the frozen split")

        detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
        recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
        calibrator_session = _cpu_session(REPO_ROOT / config["base_onnx_path"])
        detector_input = detector_session.get_inputs()[0].name
        recognizer_input = recognizer_session.get_inputs()[0].name
        calibrator_input = calibrator_session.get_inputs()[0].name
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

        phase = "floor_free_visible_selection"
        validation_values, _, records, evidence = extract_features(
            scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode="train",
            negative_cap_per_scene=int(config["validation_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        for key in (
            "scene_count",
            "proposal_count",
            "positive_proposal_count",
            "negative_proposal_count",
        ):
            if evidence[key] != config[key]:
                raise RuntimeError(f"OCR V20 P2 floor-free proposal count changed: {key}")
        if evidence["proposal_count"] != (
            registered["truth_region_count"] + registered["negative_proposal_count"]
        ):
            raise RuntimeError("OCR V20 P2 did not execute every frozen validation proposal")

        checkpoint = torch.load(
            REPO_ROOT / config["base_checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )
        model = MarginSeparatedProposalCalibrator(seed=int(config["base_model_seed"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        parity_values = torch.from_numpy(validation_values[:256])
        with torch.inference_mode():
            expected_logits = model(parity_values).numpy()
        parity_logits = np.asarray(
            calibrator_session.run(
                None,
                {calibrator_input: np.ascontiguousarray(parity_values.numpy())},
            )[0],
            dtype=np.float32,
        )
        parity_error = float(np.max(np.abs(expected_logits - parity_logits)))
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        validation_logits = np.asarray(
            calibrator_session.run(
                None,
                {calibrator_input: np.ascontiguousarray(validation_values)},
            )[0],
            dtype=np.float32,
        )
        evidence["calibrator_input_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_values).tobytes(order="C")
        ).hexdigest()
        evidence["calibrator_output_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_logits).tobytes(order="C")
        ).hexdigest()
        evidence["calibrator_onnx_sha256"] = sha256_file(
            REPO_ROOT / config["base_onnx_path"]
        )
        evidence["detector_prefilter_applied"] = False
        comparisons = evaluate_thresholds(
            scenes,
            records,
            validation_logits,
            tuple(float(value) for value in config["selection_thresholds"]),
            evidence,
        )
        robust = select_robust_window(comparisons)
        selected = robust[0] if robust else max(
            comparisons,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                -item["metrics"]["false_negatives"],
                item["metrics"]["recognition_exact"],
            ),
        )
        window = robust[1] if robust else ()
        passed = robust is not None and parity_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-margin-calibrator-candidate.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection",
            "selection_gate_passed": passed,
            "production_approval": False,
            "release_eligible": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "p1_case_details_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_aggregate_metrics_only_used_for_design": True,
            "isolated_change": config["isolated_change"],
            "optimizer_steps": 0,
            "weights_changed": False,
            "base_candidate_id": "P1",
            "base_checkpoint_path": config["base_checkpoint_path"],
            "base_checkpoint_sha256": config["base_checkpoint_sha256"],
            "onnx_path": config["base_onnx_path"],
            "onnx_sha256": config["base_onnx_sha256"],
            "onnx_parity_maximum_absolute_error": parity_error,
            "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window),
            "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"],
            "case_level_details_emitted": False,
            "public_gate_archive_opened": False,
            "public_gate_evaluations": 0,
            "marker_creation_evaluated": False,
            "training_authorization": authorization.binding,
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
            "schema": "graphreader.ocr-margin-calibrator-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "optimizer_steps": 0,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "production_approval": False,
            "release_eligible": False,
            "public_gate_evaluations": 0,
            "public_gate_archive_opened": False,
            "training_authorization": authorization.binding,
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
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = evaluate_candidate(REPO_ROOT / args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "optimizer_steps": report["optimizer_steps"],
                "selected_threshold": report["selected_threshold"],
                "passing_threshold_window": report["passing_threshold_window"],
                "selection_metrics": report["selection_metrics"],
                "onnx_parity_maximum_absolute_error": report[
                    "onnx_parity_maximum_absolute_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
