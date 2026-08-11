# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Run the first normalized-input marker-center training candidate and gate."""

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

from ml.markers.center.background_invariant_v3.pipeline import (
    MINIMUM_CENTER_SEPARATION,
    POSTPROCESS_REVISION,
    PREPROCESS_REVISION,
    evaluate_scenes,
    extract_background_invariant_proposals,
    normalize_proposal_patches,
)
from ml.markers.center.line_aware_v1.model import candidate_loss
from ml.markers.center.line_aware_v1.pipeline import (
    TrainingExamples,
    concatenate_examples,
    sample_training_examples,
)
from ml.markers.center.normalized_training_v4.dataset import (
    build_selection_scenes,
    selection_manifest,
)
from ml.markers.center.normalized_training_v4.public_gate import evaluate_candidate
from ml.markers.center.radial_feature_v1.model import RadialFeatureModelConfig, RadialFeatureNet
from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-normalized-training-v4"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/normalized_training_v4/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/normalized_training_v4/dataset.py"),
    Path("ml/markers/center/normalized_training_v4/candidate_runner.py"),
    Path("ml/markers/center/normalized_training_v4/public_gate.py"),
    Path("ml/markers/center/background_invariant_v3/pipeline.py"),
    Path("ml/markers/center/runtime_consistency_v2/pipeline_p2.py"),
    Path("ml/markers/center/radial_feature_v1/model.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/line_aware_v1/model.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _configure(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _export(model: RadialFeatureNet, sample: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model,
        sample,
        path,
        input_names=[model.contract.input_name],
        output_names=[model.contract.output_name],
        opset_version=18,
        dynamic_axes={
            model.contract.input_name: {0: "candidate_count"},
            model.contract.output_name: {0: "candidate_count"},
        },
        dynamo=False,
    )


def _runner(model: RadialFeatureNet):
    def run(value: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(value)).numpy()

    return run


def _normalize_examples(examples: TrainingExamples) -> TrainingExamples:
    return TrainingExamples(
        normalize_proposal_patches(examples.patches),
        examples.labels,
        examples.offsets,
        examples.radii,
    )


def run_candidate(output_dir: Path) -> dict[str, object]:
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
    optimizer_steps = 0
    training_seal_completed = False
    try:
        seed = int(config["seed"])
        _configure(seed)
        if config["preprocess_revision"] != PREPROCESS_REVISION:
            raise RuntimeError("P1 preprocessing differs from preregistration")
        if config["postprocess_revision"] != POSTPROCESS_REVISION:
            raise RuntimeError("P1 postprocessing differs from preregistration")
        if float(config["minimum_center_separation"]) != MINIMUM_CENTER_SEPARATION:
            raise RuntimeError("P1 center separation differs from preregistration")
        selection_path = REPO_ROOT / config["selection_manifest_path"]
        if sha256_file(selection_path) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection manifest checksum does not match preregistration")
        if _canonical_hash(selection_manifest()) != config["selection_manifest_sha256"]:
            raise RuntimeError("Selection renderer no longer reproduces the frozen manifest")
        sealed_path = REPO_ROOT / config["sealed_public_test_seal_path"]
        if sha256_file(sealed_path) != config["sealed_public_test_seal_sha256"]:
            raise RuntimeError("Sealed public test checksum does not match preregistration")
        sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
        archive = REPO_ROOT / sealed["fixture_archive_path"]
        if sha256_file(archive) != sealed["fixture_archive_sha256"]:
            raise RuntimeError("Sealed public archive changed before training")

        generator = torch.Generator().manual_seed(seed + 1)
        training_scenes = build_selection_scenes("train")
        validation_scenes = build_selection_scenes("validation")
        examples = concatenate_examples(
            _normalize_examples(
                sample_training_examples(
                    scene,
                    maximum_negative_per_positive=int(config["maximum_negative_per_positive"]),
                    generator=generator,
                )
            )
            for scene in training_scenes
        )
        model = RadialFeatureNet(RadialFeatureModelConfig(seed=seed))
        phase = "onnx_preflight"
        preflight_path = output_dir / "export-preflight.onnx"
        _export(model.eval(), examples.patches[:8], preflight_path)
        preflight_sha256 = sha256_file(preflight_path)
        preflight_path.unlink()

        phase = "training"
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config["weight_decay"]),
        )
        batch_size = int(config["batch_size"])
        checkpoints: list[dict[str, float | int]] = []
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(examples.labels), generator=generator)
            losses: list[dict[str, float]] = []
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                raw = model.forward_raw(examples.patches.index_select(0, indices))
                loss, components = candidate_loss(
                    raw,
                    examples.labels.index_select(0, indices),
                    examples.offsets.index_select(0, indices),
                    examples.radii.index_select(0, indices),
                    positive_weight=float(config["positive_loss_weight"]),
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(components)
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append(
                    {
                        "epoch": epoch + 1,
                        **{
                            key: sum(item[key] for item in losses) / len(losses)
                            for key in ("total", "marker", "offset", "radius")
                        },
                    }
                )

        phase = "selection"
        model.eval()
        threshold_results = [
            {
                "threshold": float(threshold),
                "metrics": evaluate_scenes(
                    validation_scenes,
                    _runner(model),
                    threshold=float(threshold),
                ),
            }
            for threshold in config["selection_thresholds"]
        ]
        selected = max(
            threshold_results,
            key=lambda item: (
                item["metrics"]["exact_scene_count"],
                -item["metrics"]["false_positives"],
                item["metrics"]["f1"],
                item["threshold"],
            ),
        )
        metrics = selected["metrics"]
        selection_passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["false_positives"] == 0
            and metrics["false_negatives"] == 0
            and metrics["duplicate_count"] == 0
            and metrics["prohibited_structure_hits"] == 0
        )

        phase = "export"
        checkpoint_path = output_dir / "marker-center-normalized-training-p1.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": model.export_contract(),
                "preprocess_revision": PREPROCESS_REVISION,
                "selected_threshold": selected["threshold"],
            },
            checkpoint_path,
        )
        onnx_path = output_dir / "marker-center-normalized-training-p1.onnx"
        parity_patches = extract_background_invariant_proposals(validation_scenes[0].tensor).patches[:128]
        _export(model, parity_patches, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("P1 parity requires CPUExecutionProvider only")
        parity_input = parity_patches.numpy().astype(np.float32, copy=False)
        with torch.inference_mode():
            expected = model(parity_patches).numpy()
        actual = session.run(None, {model.contract.input_name: parity_input})[0]
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= float(config["onnx_parity_tolerance"])

        report: dict[str, object] = {
            "schema": "graphreader.marker-center-normalized-training-report-p1.v4",
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
            "training_example_count": len(examples.labels),
            "epochs": config["epochs"],
            "seed": seed,
            "optimizer_steps": optimizer_steps,
            "weights_changed": True,
            "loss_checkpoints": checkpoints,
            "onnx_preflight_sha256": preflight_sha256,
            "threshold_comparisons": threshold_results,
            "selected_threshold": selected["threshold"],
            "selection_metrics": metrics,
            "selection_gate_passed": selection_passed,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "onnx_provider": "CPUExecutionProvider",
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_tolerance": config["onnx_parity_tolerance"],
            "onnx_parity_passed": parity_passed,
            "tensor_contract": model.export_contract(),
            "preprocess_revision": PREPROCESS_REVISION,
            "postprocess_revision": POSTPROCESS_REVISION,
            "minimum_center_separation": MINIMUM_CENTER_SEPARATION,
            "selection_manifest_sha256": config["selection_manifest_sha256"],
            "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
            "sealed_public_archive_opened": False,
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
            "schema": "graphreader.marker-center-normalized-training-run-summary-p1.v4",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "candidate_status": report["status"],
            "candidate_report_path": report_path.relative_to(REPO_ROOT).as_posix(),
            "candidate_report_sha256": sha256_file(report_path),
            "public_gate_executed": public_report is not None,
            "public_gate_status": None if public_report is None else public_report["status"],
            "public_gate_report_path": None if public_report is None else public_report_path.relative_to(REPO_ROOT).as_posix(),
            "public_gate_report_sha256": None if public_report is None else sha256_file(public_report_path),
            "production_approval": False,
            "release_eligible": False,
            "rerun_allowed": False,
        }
        summary_path.write_bytes(canonical_json_bytes(summary))
        return summary
    except Exception as error:
        failure = {
            "schema": "graphreader.marker-center-normalized-training-failure-p1.v4",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "release_eligible": False,
            "production_approval": False,
            "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False,
            "phase": phase,
            "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_authorization": authorization.binding,
        }
        if not report_path.exists():
            report_path.write_bytes(canonical_json_bytes(failure))
        if not training_seal_completed:
            complete_training_candidate(
                authorization,
                status="failed_runner",
                report_sha256=sha256_file(report_path),
            )
        summary_path.write_bytes(canonical_json_bytes(failure))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/markers/center/artifacts/normalized-training-v4/P1-run"),
    )
    args = parser.parse_args()
    result = run_candidate(REPO_ROOT / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("public_gate_status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
