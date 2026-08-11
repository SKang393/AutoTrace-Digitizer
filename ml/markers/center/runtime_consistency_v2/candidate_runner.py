# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""One-time zero-training validation of the exact radial P3 runtime payload."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.center.line_aware_v1.pipeline import extract_proposals
from ml.markers.center.radial_feature_v1.model import (
    RadialFeatureModelConfig,
    RadialFeatureNet,
)
from ml.markers.center.runtime_consistency_v2.dataset import (
    build_selection_scenes,
    selection_manifest,
)
from ml.markers.center.runtime_consistency_v2.pipeline import (
    POSTPROCESS_REVISION,
    evaluate_scenes,
)
from ml.markers.center.runtime_consistency_v2.public_gate import evaluate_candidate
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import (
    acquire_training_candidate,
    complete_training_candidate,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-runtime-consistency-v2"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/runtime_consistency_v2/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/runtime_consistency_v2/dataset.py"),
    Path("ml/markers/center/runtime_consistency_v2/pipeline.py"),
    Path("ml/markers/center/runtime_consistency_v2/candidate_runner.py"),
    Path("ml/markers/center/runtime_consistency_v2/public_gate.py"),
    Path("ml/markers/center/radial_feature_v1/model.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/line_aware_v1/model.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_exact_model(checkpoint_path: Path, seed: int) -> RadialFeatureNet:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RadialFeatureNet(RadialFeatureModelConfig(seed=seed))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def validate_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
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
    public_report_path = output_dir / "public-gate-report.json"
    summary_path = output_dir / "run-summary.json"
    started = time.perf_counter()
    phase = "initialization"
    training_seal_completed = False
    try:
        if int(config["optimizer_steps"]) != 0 or config["weights_changed"] is not False:
            raise RuntimeError("Runtime-consistency candidate must perform zero training")
        if config["postprocess_revision"] != POSTPROCESS_REVISION:
            raise RuntimeError("Postprocess revision differs from the frozen runtime")
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection manifest checksum does not match preregistration")
        if _canonical_hash(selection_manifest()) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection renderer no longer reproduces the frozen manifest")
        seal_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(seal_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Sealed public test checksum does not match preregistration")
        sealed = json.loads(seal_path.read_text(encoding="utf-8"))
        archive = REPO_ROOT / sealed["fixture_archive_path"]
        if sha256_file(archive) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("Sealed public archive changed before candidate execution")

        source_report_path = REPO_ROOT / config["source_training_report_path"]
        source_checkpoint_path = REPO_ROOT / config["source_checkpoint_path"]
        source_onnx_path = REPO_ROOT / config["source_onnx_path"]
        for path, expected in (
            (source_report_path, config["source_training_report_sha256"]),
            (source_checkpoint_path, config["source_checkpoint_sha256"]),
            (source_onnx_path, config["source_onnx_sha256"]),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(f"Exact source payload drifted: {path}")
        source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
        if (
            source_report.get("status") != "selected"
            or source_report.get("optimizer_steps") != 0
            or source_report.get("weights_changed") is not False
            or source_report.get("postprocess_revision") != POSTPROCESS_REVISION
            or float(source_report.get("selected_threshold", -1.0))
            != float(config["selected_threshold"])
        ):
            raise RuntimeError("Source report does not bind the exact radial P3 runtime")

        checkpoint_path = output_dir / "marker-center-runtime-consistency-p1.pt"
        onnx_path = output_dir / "marker-center-runtime-consistency-p1.onnx"
        shutil.copyfile(source_checkpoint_path, checkpoint_path)
        shutil.copyfile(source_onnx_path, onnx_path)
        if sha256_file(checkpoint_path) != config["source_checkpoint_sha256"]:
            raise RuntimeError("Copied checkpoint bytes differ from the exact source")
        if sha256_file(onnx_path) != config["source_onnx_sha256"]:
            raise RuntimeError("Copied ONNX bytes differ from the exact source")

        phase = "selection"
        validation_scenes = build_selection_scenes("validation")
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Candidate validation requires CPUExecutionProvider only")
        input_name = session.get_inputs()[0].name

        def onnx_runner(value: np.ndarray) -> np.ndarray:
            return session.run(None, {input_name: value})[0]

        selected_threshold = float(config["selected_threshold"])
        metrics = evaluate_scenes(
            validation_scenes,
            onnx_runner,
            threshold=selected_threshold,
        )
        selection_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )

        phase = "parity"
        model = _load_exact_model(
            checkpoint_path,
            int(config["source_model_seed"]),
        )
        parity_patches = extract_proposals(validation_scenes[0].tensor).patches[:128]
        parity_input = parity_patches.numpy().astype(np.float32, copy=False)
        with torch.inference_mode():
            expected = model(parity_patches).numpy()
        actual = onnx_runner(parity_input)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])

        report: dict[str, object] = {
            "schema": "graphreader.marker-center-runtime-consistency-candidate.v2",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "selected" if selection_passed and parity_passed else "failed_selection",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "training_authorization": authorization.binding,
            "optimizer_steps": 0,
            "weights_changed": False,
            "source_training_report_sha256": config["source_training_report_sha256"],
            "source_checkpoint_sha256": config["source_checkpoint_sha256"],
            "source_onnx_sha256": config["source_onnx_sha256"],
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "postprocess_revision": POSTPROCESS_REVISION,
            "selected_threshold": selected_threshold,
            "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
            "public_gate_authorized_on_selection_pass": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(
            authorization,
            status=str(report["status"]),
            report_sha256=sha256_file(report_path),
        )
        training_seal_completed = True

        public_report: dict[str, object] | None = None
        if report["status"] == "selected":
            phase = "public_gate"
            public_report = evaluate_candidate(
                onnx_path=onnx_path,
                candidate_report_path=report_path,
                output_path=public_report_path,
            )
        summary = {
            "schema": "graphreader.marker-center-runtime-consistency-run-summary.v2",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "candidate_status": report["status"],
            "candidate_report_path": report_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_report_sha256": sha256_file(report_path),
            "public_gate_executed": public_report is not None,
            "public_gate_status": None if public_report is None else public_report["status"],
            "public_gate_report_path": (
                None
                if public_report is None
                else public_report_path.relative_to(REPO_ROOT).as_posix()
            ),
            "public_gate_report_sha256": (
                None if public_report is None else sha256_file(public_report_path)
            ),
            "production_approval": False,
            "release_eligible": False,
        }
        summary_path.write_bytes(canonical_json_bytes(summary))
        return summary
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-runtime-consistency-failure.v2",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "sealed_public_archive_opened": phase == "public_gate",
            "phase": phase,
            "optimizer_steps": 0,
            "weights_changed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
        }
        failure_path = (
            output_dir / "public-gate-failure.json"
            if training_seal_completed
            else report_path
        )
        if training_seal_completed:
            failure["candidate_report_sha256"] = sha256_file(report_path)
        failure_path.write_bytes(canonical_json_bytes(failure))
        if not training_seal_completed:
            complete_training_candidate(
                authorization,
                status="failed_runner",
                report_sha256=sha256_file(report_path),
            )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ml/markers/center/artifacts/runtime-consistency-v2/P1-run"
        ),
    )
    args = parser.parse_args()
    result = validate_candidate(REPO_ROOT / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["public_gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
