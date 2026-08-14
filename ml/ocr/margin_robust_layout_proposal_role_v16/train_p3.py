# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use proposal-only hard-negative repair for OCR V16 P3."""

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
from .dataset import build_split, split_fingerprint, training_examples
from .model import MarginRobustLayoutProposalRoleNet
from .model_p3 import OutputScaledMarginCandidate
from .pipeline import evaluate_thresholds
from .protocol import ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION, TASK
from .train_p1 import _export, _select_robust_window


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/margin_robust_layout_proposal_role_v16")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "P1_RESULT.json", ROOT / "P2_RESULT.json", ROOT / "dataset.py", ROOT / "model.py",
    ROOT / "model_p3.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "train_p1.py",
    ROOT / "train_p3.py", Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
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


def _hard_negative_indices(
    model: nn.Module,
    values: torch.Tensor,
    labels: torch.Tensor,
    *,
    multiplier: int,
    batch_size: int,
) -> torch.Tensor:
    positive = torch.nonzero(labels == 1, as_tuple=False).flatten()
    negative = torch.nonzero(labels == 0, as_tuple=False).flatten()
    if len(positive) == 0 or len(negative) == 0:
        raise RuntimeError("OCR V16 P3 hard-negative mining requires both proposal classes")
    scores: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(negative), batch_size):
            indices = negative[start:start + batch_size]
            output = model(values.index_select(0, indices))[:, :2]
            scores.append(torch.softmax(output, dim=1)[:, 1])
    score = torch.cat(scores)
    count = min(len(negative), len(positive) * multiplier)
    order = torch.argsort(score, descending=True, stable=True)[:count]
    return negative.index_select(0, order)


def _proposal_only_parameters(model: MarginRobustLayoutProposalRoleNet) -> list[nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = (
        model.base.base.proposal_head,
        model.base.base.proposal_residual,
        model.layout_proposal,
    )
    values: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            values.append(parameter)
    if not values:
        raise RuntimeError("OCR V16 P3 proposal-only parameter set is empty")
    return values


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V16 P3 output exists: {output_dir}")
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
            raise RuntimeError("OCR V16 P3 runner sources changed")
        p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
        p2 = json.loads((REPO_ROOT / P2_RESULT_PATH).read_text(encoding="utf-8"))
        if sha256_file(REPO_ROOT / P1_RESULT_PATH) != config["p1_result_sha256"]:
            raise RuntimeError("OCR V16 P1 result changed")
        if sha256_file(REPO_ROOT / P2_RESULT_PATH) != config["p2_result_sha256"]:
            raise RuntimeError("OCR V16 P2 result changed")
        checkpoint_path = REPO_ROOT / p1["checkpoint_path"]
        if (
            sha256_file(checkpoint_path) != p1["checkpoint_sha256"]
            or p1["checkpoint_sha256"] != config["p1_checkpoint_sha256"]
        ):
            raise RuntimeError("OCR V16 P1 checkpoint changed")
        if (
            p1["candidate_report_sha256"] != config["p1_candidate_report_sha256"]
            or p2["candidate_report_sha256"] != config["p2_candidate_report_sha256"]
        ):
            raise RuntimeError("OCR V16 consumed candidate reports changed")
        if p1["status"] != p2["status"] or p1["status"] != "failed_selection":
            raise RuntimeError("OCR V16 P3 requires exact consumed P1 and P2 selection failures")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        train, validation = build_split("train"), build_split("validation")
        if split_fingerprint(train) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V16 P3 train split changed")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V16 P3 validation split changed")
        values, proposal_labels, _, training_evidence = training_examples(
            train, negative_cap_per_scene=int(config["negative_cap_per_scene"]),
        )
        if any(training_evidence[key] is not False for key in (
            "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
            "v15_public_fixture_bytes_scene_truth_or_case_identity_used",
        )):
            raise RuntimeError("OCR V16 P3 training data violated preregistered scope")
        for key in (
            "proposal_count", "positive_proposal_count", "negative_proposal_count",
            "tensor_label_stream_sha256",
        ):
            if training_evidence[key] != config[key]:
                raise RuntimeError(f"OCR V16 P3 training evidence changed: {key}")

        generator = _configure(int(config["seed"]))
        base = MarginRobustLayoutProposalRoleNet(seed=int(config["seed"]))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base.load_state_dict(checkpoint["state_dict"], strict=True)
        tensors = torch.from_numpy(values)
        targets = torch.from_numpy(proposal_labels)
        parity_values = tensors[:256]
        base.eval()
        with torch.inference_mode():
            roles_before = base(parity_values)[:, 2:].clone()
        hard_negatives = _hard_negative_indices(
            base, tensors, targets,
            multiplier=int(config["hard_negative_multiplier"]),
            batch_size=int(config["mining_batch_size"]),
        )
        if len(hard_negatives) != int(config["expected_hard_negative_count"]):
            raise RuntimeError("OCR V16 P3 hard-negative count changed")
        positives = torch.nonzero(targets == 1, as_tuple=False).flatten()
        selected_indices = torch.cat((positives, hard_negatives))
        trainable = _proposal_only_parameters(base)
        optimizer = torch.optim.AdamW(
            trainable, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        class_weights = torch.tensor((float(config["negative_class_weight"]), 1.0), dtype=torch.float32)
        proposal_criterion = nn.CrossEntropyLoss(weight=class_weights)
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        base.train()
        for epoch in range(int(config["epochs"])):
            order = selected_indices.index_select(
                0, torch.randperm(len(selected_indices), generator=generator),
            )
            losses: list[float] = []
            margins: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                batch_targets = targets.index_select(0, indices)
                output = base(tensors.index_select(0, indices))[:, :2]
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
            raise RuntimeError("OCR V16 P3 optimizer-step count changed")

        base.eval()
        with torch.inference_mode():
            roles_after = base(parity_values)[:, 2:]
        role_invariance_error = float(torch.max(torch.abs(roles_before - roles_after)))
        if role_invariance_error != 0.0:
            raise RuntimeError("OCR V16 P3 changed frozen role logits")
        checkpoint_output = output_dir / "graph-text-margin-robust-layout-proposal-role-v16-p3.pt"
        torch.save({"state_dict": base.state_dict()}, checkpoint_output)
        model = OutputScaledMarginCandidate(base, output_scale=float(config["output_scale"])).eval()
        phase = "export"
        onnx_path = output_dir / "graph-text-margin-robust-layout-proposal-role-v16-p3.onnx"
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V16 P3 selection requires CPU execution only")
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
        passed = robust is not None and parity_passed and role_argmax_mismatches == 0 and calls == len(validation)
        report: dict[str, object] = {
            "schema": "graphreader.ocr-margin-robust-layout-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v15_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_p2_aggregate_metrics_only_used_for_design": True,
            "p1_p2_validation_case_detail_or_pixels_used_for_design": False,
            "p1_checkpoint_path": p1["checkpoint_path"], "p1_checkpoint_sha256": p1["checkpoint_sha256"],
            "optimizer_steps": optimizer_steps, "weights_changed": True,
            "hard_negative_count": int(len(hard_negatives)),
            "hard_negative_multiplier": config["hard_negative_multiplier"],
            "output_scale": config["output_scale"],
            "role_invariance_maximum_absolute_error": role_invariance_error,
            "training_evidence": training_evidence, "loss_checkpoints": checkpoints,
            "checkpoint_path": checkpoint_output.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_output),
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
        "hard_negative_count": report["hard_negative_count"], "output_scale": report["output_scale"],
        "selected_threshold": report["selected_threshold"],
        "passing_threshold_window": report["passing_threshold_window"],
        "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
