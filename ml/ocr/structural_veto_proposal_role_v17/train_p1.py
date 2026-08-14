# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 structural-veto training for OCR V17."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from ml.ocr.margin_robust_layout_proposal_role_v16.train_p1 import _export, _select_robust_window
from .dataset import build_split, proposal_summary, split_fingerprint, training_examples
from .model import StructuralVetoProposalRoleNet
from .pipeline import evaluate_thresholds
from .protocol import BASE_CHECKPOINT_SHA256, ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/structural_veto_proposal_role_v17")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "pipeline.py", ROOT / "protocol.py",
    ROOT / "train_p1.py", Path("ml/ocr/margin_robust_layout_proposal_role_v16/model.py"),
    Path("ml/ocr/margin_robust_layout_proposal_role_v16/pipeline.py"),
    Path("ml/ocr/margin_robust_layout_proposal_role_v16/train_p1.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model_p2.py"),
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


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V17 P1 output exists: {output_dir}")
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
            raise RuntimeError("OCR V17 P1 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V17 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V17 public seal changed")
        base_path = REPO_ROOT / config["base_checkpoint_path"]
        if sha256_file(base_path) != BASE_CHECKPOINT_SHA256 or config["base_checkpoint_sha256"] != BASE_CHECKPOINT_SHA256:
            raise RuntimeError("OCR V17 base checkpoint changed")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        train, validation = build_split("train"), build_split("validation")
        if split_fingerprint(train) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V17 train split changed")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V17 validation split changed")
        summary = proposal_summary(train)
        if summary != {key: selection["train"][key] for key in summary}:
            raise RuntimeError("OCR V17 train proposals changed")
        values, proposal_labels, _, training_evidence = training_examples(
            train, negative_cap_per_scene=int(config["negative_cap_per_scene"]),
        )
        for key in (
            "scene_count", "negative_cap_per_scene", "negative_sampling", "negative_family_counts",
            "proposal_count", "positive_proposal_count", "negative_proposal_count", "tensor_label_stream_sha256",
        ):
            if training_evidence[key] != config[key]:
                raise RuntimeError(f"OCR V17 training evidence changed: {key}")
        if any(training_evidence[key] is not False for key in (
            "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
            "v16_fixture_bytes_scene_truth_or_case_identity_used",
        )):
            raise RuntimeError("OCR V17 training data violated preregistered scope")

        generator = _configure(int(config["seed"]))
        model = StructuralVetoProposalRoleNet(
            seed=int(config["seed"]), base_output_scale=float(config["base_output_scale"]),
        )
        checkpoint = torch.load(base_path, map_location="cpu", weights_only=True)
        model.base.load_state_dict(checkpoint["state_dict"], strict=True)
        tensors = torch.from_numpy(values)
        targets = torch.from_numpy(proposal_labels)
        parity_values = tensors[:256]
        model.eval()
        with torch.inference_mode():
            roles_before = model(parity_values)[:, 2:].clone()
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        class_weights = torch.tensor((float(config["negative_class_weight"]), 1.0), dtype=torch.float32)
        proposal_criterion = nn.CrossEntropyLoss(weight=class_weights)
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        model.train()
        for epoch in range(int(config["epochs"])):
            order = torch.randperm(len(tensors), generator=generator)
            losses: list[float] = []
            margins: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                batch_targets = targets.index_select(0, indices)
                output = model(tensors.index_select(0, indices))[:, :2]
                proposal_loss = proposal_criterion(output, batch_targets)
                signed_target = batch_targets.to(torch.float32).mul(2.0).sub(1.0)
                signed_logit = output[:, 1] - output[:, 0]
                desired_margin = torch.where(
                    batch_targets == 1,
                    torch.full_like(signed_logit, float(config["positive_margin"])),
                    torch.full_like(signed_logit, float(config["negative_margin"])),
                )
                margin_loss = torch.relu(desired_margin - signed_target * signed_logit).mean()
                loss = proposal_loss + float(config["margin_loss_weight"]) * margin_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                losses.append(float(loss.detach()))
                margins.append(float(margin_loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({
                    "epoch": epoch + 1,
                    "proposal_loss": sum(losses) / len(losses),
                    "proposal_margin_loss": sum(margins) / len(margins),
                })
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V17 P1 optimizer-step count changed")
        model.eval()
        with torch.inference_mode():
            roles_after = model(parity_values)[:, 2:]
        role_invariance_error = float(torch.max(torch.abs(roles_before - roles_after)))
        if role_invariance_error != 0.0:
            raise RuntimeError("OCR V17 P1 changed frozen role logits")

        phase = "export"
        checkpoint_path = output_dir / "graph-text-structural-veto-proposal-role-v17-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-structural-veto-proposal-role-v17-p1.onnx"
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V17 selection requires CPU execution only")
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        role_argmax_mismatches = int(np.count_nonzero(
            np.argmax(expected[:, 2:], axis=1) != np.argmax(actual[:, 2:], axis=1)
        ))
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
        robust = _select_robust_window(comparisons)
        selected = robust[0] if robust else max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["role_accuracy"], item["threshold"],
        ))
        window = robust[1] if robust else ()
        passed = (
            robust is not None and parity_passed and role_argmax_mismatches == 0
            and calls == len(validation)
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-structural-veto-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v16_fixture_bytes_scene_truth_or_case_identity_used": False,
            "v16_aggregate_metrics_only_used_for_design": True,
            "optimizer_steps": optimizer_steps, "training_evidence": training_evidence,
            "loss_checkpoints": checkpoints,
            "base_checkpoint_path": config["base_checkpoint_path"],
            "base_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
            "base_parameters_frozen": True,
            "role_invariance_maximum_absolute_error": role_invariance_error,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
            "schema": "graphreader.ocr-structural-veto-proposal-role-failure.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "failed_runner", "phase": phase, "optimizer_steps": optimizer_steps,
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
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
