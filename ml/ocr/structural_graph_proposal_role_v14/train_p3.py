# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use final head-only repair for structural graph OCR V14 P3."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, proposal_summary, split_fingerprint, training_examples
from .model_p3 import OutputScaledCandidate, StructuralGraphProposalRoleP3Net
from .pipeline import evaluate_thresholds
from .protocol import (
    ENCODED_WIDTH, ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION,
    ROLE_ACCURACY_MINIMUM, ROLE_CLASS_ACCURACY_MINIMUM, ROLE_ORDER, TASK,
)
from .train_p1 import _balanced_order, _configure


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/structural_graph_proposal_role_v14")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
P1_CHECKPOINT_PATH = ROOT / "artifacts/P1-run/graph-text-structural-graph-proposal-role-v14-p1.pt"
P2_REPORT_PATH = ROOT / "artifacts/P2-run/candidate-report.json"
P2_ONNX_PATH = ROOT / "artifacts/P2-run/graph-text-structural-graph-proposal-role-v14-p2.onnx"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "model_p2.py", ROOT / "model_p3.py",
    ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "train_p1.py", ROOT / "train_p3.py",
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model, example, path,
        input_names=["region_proposals"], output_names=["proposal_role_logits"],
        dynamic_axes={"region_proposals": {0: "proposal_count"}, "proposal_role_logits": {0: "proposal_count"}},
        opset_version=18, dynamo=False,
    )


def _load_trigger(config: dict[str, object]) -> dict[str, object]:
    if sha256_file(REPO_ROOT / P1_RESULT_PATH) != config["p1_result_sha256"]:
        raise RuntimeError("OCR V14 P1 result changed before P3")
    if sha256_file(REPO_ROOT / P2_RESULT_PATH) != config["p2_result_sha256"]:
        raise RuntimeError("OCR V14 P2 result changed before P3")
    p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
    p2 = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
    if (
        p1.get("status") != "failed_runner" or p1.get("consumed") is not True
        or p1.get("public_gate_archive_opened") is not False or p1.get("public_gate_evaluations") != 0
        or p2.get("status") != "failed_selection" or p2.get("consumed") is not True
        or p2.get("public_gate_archive_opened") is not False or p2.get("public_gate_evaluations") != 0
    ):
        raise RuntimeError("OCR V14 P3 trigger candidates are not exact consumed unopened evidence")
    if (
        p1["checkpoint_sha256"] != config["p1_checkpoint_sha256"]
        or sha256_file(REPO_ROOT / P1_CHECKPOINT_PATH) != config["p1_checkpoint_sha256"]
        or p2["candidate_report_sha256"] != config["p2_report_sha256"]
        or sha256_file(REPO_ROOT / P2_REPORT_PATH) != config["p2_report_sha256"]
        or p2["onnx_sha256"] != config["p2_onnx_sha256"]
        or sha256_file(REPO_ROOT / P2_ONNX_PATH) != config["p2_onnx_sha256"]
    ):
        raise RuntimeError("OCR V14 P3 trigger payload binding changed")
    return p2


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V14 P3 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started, phase, optimizer_steps = time.perf_counter(), "initialization", 0
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V14 P3 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V14 P3 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V14 P3 public seal changed")
        p2 = _load_trigger(config)
        if (
            p2["selection_metrics"]["true_positives"] != config["p2_aggregate_trigger"]["true_positives"]
            or p2["selection_metrics"]["false_positives"] != config["p2_aggregate_trigger"]["false_positives"]
            or p2["selection_metrics"]["false_negatives"] != config["p2_aggregate_trigger"]["false_negatives"]
            or p2["selection_metrics"]["role_accuracy"] != config["p2_aggregate_trigger"]["role_accuracy"]
            or p2["selection_metrics"]["per_role_accuracy"]["PhaseHeading"]
            != config["p2_aggregate_trigger"]["phase_heading_accuracy"]
            or p2["selection_metrics"]["per_role_accuracy"]["YTick"]
            != config["p2_aggregate_trigger"]["y_tick_accuracy"]
        ):
            raise RuntimeError("OCR V14 P3 aggregate-only trigger changed")
        seal = json.loads((REPO_ROOT / SEAL_PATH).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / seal["fixture_archive_path"]) != seal["fixture_archive_sha256"]:
            raise RuntimeError("OCR V14 sealed-public archive changed before P3")

        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        train, validation = build_split("train"), build_split("validation")
        if split_fingerprint(train) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V14 P3 train split changed")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V14 P3 validation split changed")
        if proposal_summary(train) != {key: selection["train"][key] for key in proposal_summary(train)}:
            raise RuntimeError("OCR V14 P3 train proposal evidence changed")
        values, proposal_labels, role_labels, training_evidence = training_examples(
            train, negative_cap_per_scene=int(config["negative_cap_per_scene"]),
        )
        for key in (
            "scene_count", "negative_cap_per_scene", "negative_sampling", "negative_family_counts",
            "proposal_count", "positive_proposal_count", "negative_proposal_count", "tensor_label_stream_sha256",
        ):
            if training_evidence[key] != config[key]:
                raise RuntimeError(f"OCR V14 P3 training evidence changed: {key}")
        if any(training_evidence[key] is not False for key in (
            "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
            "v13_public_fixture_bytes_scene_truth_or_case_identity_used",
        )):
            raise RuntimeError("OCR V14 P3 training data violated preregistered scope")

        generator = _configure(int(config["seed"]))
        model = StructuralGraphProposalRoleP3Net(
            base_seed=int(config["base_seed"]), residual_seed=int(config["seed"]),
        )
        state = torch.load(REPO_ROOT / P1_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state["state_dict"], strict=False)
        expected_missing = {
            "role_geometry_residual.0.weight", "role_geometry_residual.0.bias",
            "role_geometry_residual.2.weight", "role_geometry_residual.2.bias",
        }
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("OCR V14 P3 did not load the exact P1 state contract")
        wrapped = OutputScaledCandidate(model.eval(), float(config["output_scale"]))
        phase = "onnx_preflight"
        preflight_path = output_dir / "export-preflight.onnx"
        _export(wrapped, torch.zeros((8, 2, 32, ENCODED_WIDTH), dtype=torch.float32), preflight_path)
        preflight_sha256 = sha256_file(preflight_path)
        preflight_session = ort.InferenceSession(str(preflight_path), providers=["CPUExecutionProvider"])
        preflight_output = np.asarray(
            preflight_session.run(
                None, {"region_proposals": np.zeros((3, 2, 32, ENCODED_WIDTH), dtype=np.float32)},
            )[0], dtype=np.float32,
        )
        if preflight_output.shape != (3, 10) or not np.isfinite(preflight_output).all():
            raise RuntimeError("OCR V14 P3 export preflight returned invalid logits")
        preflight_path.unlink()

        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainable_modules = (
            model.proposal_head, model.proposal_residual, model.role_head, model.role_geometry_residual,
        )
        for module in trainable_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        trainable_names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if trainable_names != sorted(config["trainable_parameter_names"]):
            raise RuntimeError("OCR V14 P3 trainable parameter boundary changed")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        proposal_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([float(config["negative_class_weight"]), 1.0], dtype=torch.float32),
        )
        role_criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([float(config["role_class_weights"][role]) for role in ROLE_ORDER]),
        )
        tensors = torch.from_numpy(values)
        proposal_targets = torch.from_numpy(proposal_labels)
        role_targets = torch.from_numpy(role_labels)
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        model.train()
        for epoch in range(int(config["epochs"])):
            order = _balanced_order(proposal_targets, generator)
            proposal_losses: list[float] = []
            role_losses: list[float] = []
            total_losses: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                output = model(tensors.index_select(0, indices))
                proposal_loss = proposal_criterion(output[:, :2], proposal_targets.index_select(0, indices))
                batch_roles = role_targets.index_select(0, indices)
                positive = batch_roles >= 0
                role_loss = role_criterion(output[positive, 2:], batch_roles[positive])
                loss = proposal_loss + float(config["role_loss_weight"]) * role_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                proposal_losses.append(float(proposal_loss.detach()))
                role_losses.append(float(role_loss.detach()))
                total_losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({
                    "epoch": epoch + 1,
                    "proposal_loss": sum(proposal_losses) / len(proposal_losses),
                    "role_loss": sum(role_losses) / len(role_losses),
                    "total_loss": sum(total_losses) / len(total_losses),
                })
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V14 P3 optimizer-step count changed")

        phase = "export"
        model.eval()
        wrapped = OutputScaledCandidate(model, float(config["output_scale"])).eval()
        checkpoint_path = output_dir / "graph-text-structural-graph-proposal-role-v14-p3.pt"
        torch.save({"state_dict": model.state_dict(), "output_scale": config["output_scale"]}, checkpoint_path)
        onnx_path = output_dir / "graph-text-structural-graph-proposal-role-v14-p3.onnx"
        parity_values = torch.from_numpy(values[:256])
        _export(wrapped, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V14 P3 selection requires CPU execution only")
        with torch.inference_mode():
            expected = wrapped(parity_values).numpy()
        actual = np.asarray(
            session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32,
        )
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        calls = 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.asarray(
                session.run(None, {"region_proposals": np.ascontiguousarray(input_values, dtype=np.float32)})[0],
                dtype=np.float32,
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
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-structural-graph-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v13_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p2_validation_case_detail_or_pixels_used_for_design": False,
            "p2_aggregate_metrics_only_used_for_design": True,
            "optimizer_steps": optimizer_steps, "weights_changed": True,
            "isolated_change": config["isolated_change"], "training_evidence": training_evidence,
            "trainable_parameter_names": trainable_names,
            "negative_class_weight": config["negative_class_weight"],
            "role_class_weights": config["role_class_weights"],
            "role_loss_weight": config["role_loss_weight"], "loss_checkpoints": checkpoints,
            "p1_checkpoint_sha256": config["p1_checkpoint_sha256"],
            "p2_result_sha256": config["p2_result_sha256"],
            "output_scale": config["output_scale"], "preflight_sha256": preflight_sha256,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path),
            "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_passed": parity_passed, "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons, "selected_threshold": selected["threshold"],
            "selection_metrics": metrics, "direct_execution_inference_calls": calls,
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-structural-graph-proposal-role-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": optimizer_steps,
            "exception_type": type(error).__name__, "exception_message": str(error),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False,
            "public_gate_evaluations": 0, "sealed_public_archive_opened": False,
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
        "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
