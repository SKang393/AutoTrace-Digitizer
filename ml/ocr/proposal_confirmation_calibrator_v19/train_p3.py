# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 training and visible selection for OCR V19."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from .dataset import load_split_archive, proposal_summary, split_fingerprint
from .model_p3 import QuadraticProposalConfirmationCalibrator
from .pipeline import evaluate_thresholds, extract_features, select_robust_window
from .protocol import (
    DETECTOR_PATH, DETECTOR_SHA256, ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH, RECOGNIZER_SHA256, RECOGNIZER_YAML_PATH, RECOGNIZER_YAML_SHA256,
    REVISION, TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/proposal_confirmation_calibrator_v19")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model_p3.py", ROOT / "pipeline.py", ROOT / "protocol.py",
    ROOT / "train_p3.py", Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    return torch.Generator().manual_seed(seed)


def _cpu_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("OCR V19 requires CPUExecutionProvider only")
    return session


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model, example, path, input_names=["proposal_evidence"], output_names=["confirmation_logits"],
        dynamic_axes={"proposal_evidence": {0: "proposal_count"}, "confirmation_logits": {0: "proposal_count"}},
        opset_version=18, dynamo=False,
    )


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V19 P3 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started, phase, optimizer_steps = time.perf_counter(), "preflight", 0
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V19 P3 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V19 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V19 public seal changed")
        for key in ("p1", "p2"):
            result_path = Path(str(config[f"{key}_result_path"]))
            if sha256_file(REPO_ROOT / result_path) != config[f"{key}_result_sha256"]:
                raise RuntimeError(f"OCR V19 {key.upper()} aggregate result changed")
        for relative, expected in {
            DETECTOR_PATH: DETECTOR_SHA256, RECOGNIZER_PATH: RECOGNIZER_SHA256,
            RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        }.items():
            if sha256_file(REPO_ROOT / relative) != expected:
                raise RuntimeError(f"OCR V19 exact frozen input changed: {relative}")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        splits: dict[str, tuple[object, ...]] = {}
        for split in ("train", "validation"):
            registered = selection[split]
            archive = REPO_ROOT / registered["fixture_archive_path"]
            manifest = REPO_ROOT / registered["private_manifest_path"]
            if (
                sha256_file(archive) != registered["fixture_archive_sha256"]
                or sha256_file(manifest) != registered["private_manifest_sha256"]
            ):
                raise RuntimeError(f"OCR V19 stored {split} fixture bytes changed")
            scenes = load_split_archive(archive, manifest, expected_split=split)
            summary = proposal_summary(scenes)
            if (
                split_fingerprint(scenes) != registered["split_fingerprint"]
                or any(summary[key] != registered[key] for key in summary)
            ):
                raise RuntimeError(f"OCR V19 stored {split} fixtures violate the frozen split")
            splits[split] = scenes

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

        phase = "training_feature_execution"
        train_values, train_labels, _, training_evidence = extract_features(
            splits["train"], detector_runner, recognizer_runner, alphabet, mode="train",
            negative_cap_per_scene=int(config["negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        if any(training_evidence[key] != config[key] for key in (
            "scene_count", "proposal_count", "positive_proposal_count", "negative_proposal_count",
        )):
            raise RuntimeError("OCR V19 training sample counts changed")
        generator = _configure(int(config["seed"]))
        model = QuadraticProposalConfirmationCalibrator(seed=int(config["seed"]))
        tensors = torch.from_numpy(train_values)
        targets = torch.from_numpy(train_labels)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor((float(config["negative_class_weight"]), 1.0)))
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(tensors), generator=generator)
            losses: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                loss = criterion(model(tensors.index_select(0, indices)), targets.index_select(0, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({"epoch": epoch + 1, "loss": sum(losses) / len(losses)})
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V19 P3 optimizer-step count changed")

        phase = "export"
        checkpoint_path = output_dir / "graph-text-proposal-confirmation-calibrator-v19-p3.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-proposal-confirmation-calibrator-v19-p3.onnx"
        parity_values = tensors[:256]
        model.eval()
        _export(model, parity_values, onnx_path)
        calibrator_session = _cpu_session(onnx_path)
        calibrator_input = calibrator_session.get_inputs()[0].name
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(
            calibrator_session.run(None, {calibrator_input: parity_values.numpy()})[0],
            dtype=np.float32,
        )
        parity_error = float(np.max(np.abs(expected - actual)))
        parity_passed = parity_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR

        phase = "visible_selection"
        validation_values, _, records, selection_evidence = extract_features(
            splits["validation"], detector_runner, recognizer_runner, alphabet, mode="evaluate",
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        validation_logits = np.asarray(
            calibrator_session.run(None, {calibrator_input: np.ascontiguousarray(validation_values)})[0],
            dtype=np.float32,
        )
        selection_evidence["calibrator_input_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_values).tobytes(order="C")
        ).hexdigest()
        selection_evidence["calibrator_output_tensor_stream_sha256"] = hashlib.sha256(
            np.ascontiguousarray(validation_logits).tobytes(order="C")
        ).hexdigest()
        selection_evidence["calibrator_onnx_sha256"] = sha256_file(onnx_path)
        comparisons = evaluate_thresholds(
            splits["validation"], records, validation_logits,
            tuple(float(value) for value in config["selection_thresholds"]), selection_evidence,
        )
        robust = select_robust_window(comparisons)
        selected = robust[0] if robust else max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["recognition_exact"],
        ))
        window = robust[1] if robust else ()
        passed = robust is not None and parity_passed
        report: dict[str, object] = {
            "schema": "graphreader.ocr-proposal-confirmation-calibrator-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "v18_case_details_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_p2_case_details_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_p2_aggregate_metrics_only_used_for_design": True,
            "optimizer_steps": optimizer_steps, "training_evidence": training_evidence,
            "loss_checkpoints": checkpoints,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(), "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": parity_error, "onnx_parity_passed": parity_passed,
            "provider": "CPUExecutionProvider", "threshold_comparisons": comparisons,
            "passing_threshold_window": list(window), "selected_threshold": selected["threshold"],
            "selection_metrics": selected["metrics"], "case_level_details_emitted": False,
            "public_gate_archive_opened": False, "public_gate_evaluations": 0,
            "marker_creation_evaluated": False, "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-proposal-confirmation-calibrator-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False,
            "public_gate_evaluations": 0, "public_gate_archive_opened": False,
            "training_authorization": authorization.binding,
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
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
