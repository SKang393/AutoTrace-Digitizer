# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use training-only hard-negative separation repair for OCR V15 P2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from torch.nn import functional as F

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, proposal_summary, split_fingerprint, training_examples
from .model import LayoutConditionedProposalRoleNet
from .pipeline import evaluate_thresholds
from .protocol import (
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    REVISION,
    ROLE_ACCURACY_MINIMUM,
    ROLE_CLASS_ACCURACY_MINIMUM,
    TASK,
)
from .train_p1 import _configure, _export


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/layout_conditioned_proposal_role_v15")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
P1_REPORT_PATH = ROOT / "artifacts/P1-run/candidate-report.json"
P1_CHECKPOINT_PATH = ROOT / "artifacts/P1-run/graph-text-layout-conditioned-proposal-role-v15-p1.pt"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "pipeline.py", ROOT / "protocol.py",
    ROOT / "train_p1.py", ROOT / "train_p2.py",
    Path("ml/ocr/structural_graph_proposal_role_v14/model.py"),
    Path("ml/ocr/structural_graph_proposal_role_v14/model_p2.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


def _load_trigger(config: dict[str, object]) -> dict[str, object]:
    if sha256_file(REPO_ROOT / P1_RESULT_PATH) != config["p1_result_sha256"]:
        raise RuntimeError("OCR V15 P1 result changed before P2")
    result = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
    if (
        result.get("status") != "failed_selection"
        or result.get("consumed") is not True
        or result.get("public_gate_archive_opened") is not False
        or result.get("public_gate_evaluations") != 0
        or result.get("production_approval") is not False
        or result.get("release_eligible") is not False
    ):
        raise RuntimeError("OCR V15 P1 is not exact consumed unopened evidence")
    if (
        result["candidate_report_sha256"] != config["p1_report_sha256"]
        or sha256_file(REPO_ROOT / P1_REPORT_PATH) != config["p1_report_sha256"]
        or result["checkpoint_sha256"] != config["p1_checkpoint_sha256"]
        or sha256_file(REPO_ROOT / P1_CHECKPOINT_PATH) != config["p1_checkpoint_sha256"]
        or result["onnx_sha256"] != config["p1_onnx_sha256"]
    ):
        raise RuntimeError("OCR V15 P1 payload binding changed before P2")
    selected = result["selection_metrics"]
    trigger = config["p1_aggregate_trigger"]
    for key in (
        "exact_scene_count", "true_positives", "false_positives", "false_negatives",
        "duplicate_region_count", "prohibited_structure_hits", "role_accuracy",
    ):
        if selected[key] != trigger[key]:
            raise RuntimeError(f"OCR V15 P1 aggregate trigger changed: {key}")
    if result["threshold_0_88_aggregate"] != trigger["threshold_0_88"]:
        raise RuntimeError("OCR V15 P1 lower-threshold aggregate trigger changed")
    return result


def _hard_negative_indices(
    model: nn.Module,
    tensors: torch.Tensor,
    proposal_targets: torch.Tensor,
    count: int,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    margins: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), batch_size):
            logits = model(tensors[start:start + batch_size])[:, :2]
            margins.append((logits[:, 1] - logits[:, 0]).cpu())
    values = torch.cat(margins)
    negative = torch.nonzero(proposal_targets == 0, as_tuple=False).flatten()
    if count <= 0 or count > len(negative):
        raise RuntimeError("OCR V15 P2 hard-negative count is invalid")
    order = torch.argsort(values.index_select(0, negative), descending=True, stable=True)
    return negative.index_select(0, order[:count])


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V15 P2 output exists: {output_dir}")
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
            raise RuntimeError("OCR V15 P2 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V15 P2 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V15 P2 public seal changed")
        _load_trigger(config)
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        train, validation = build_split("train"), build_split("validation")
        if split_fingerprint(train) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V15 P2 train split changed")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V15 P2 validation split changed")
        if proposal_summary(train) != {key: selection["train"][key] for key in proposal_summary(train)}:
            raise RuntimeError("OCR V15 P2 train proposals changed")
        values, proposal_labels, _, training_evidence = training_examples(
            train, negative_cap_per_scene=int(config["negative_cap_per_scene"]),
        )
        for key in (
            "scene_count", "negative_cap_per_scene", "negative_sampling", "negative_family_counts",
            "proposal_count", "positive_proposal_count", "negative_proposal_count", "tensor_label_stream_sha256",
        ):
            if training_evidence[key] != config[key]:
                raise RuntimeError(f"OCR V15 P2 training evidence changed: {key}")
        if any(training_evidence[key] is not False for key in (
            "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
            "v14_validation_fixture_bytes_scene_truth_or_case_identity_used",
        )):
            raise RuntimeError("OCR V15 P2 training data violated preregistered scope")

        generator = _configure(int(config["seed"]))
        model = LayoutConditionedProposalRoleNet(seed=int(config["base_seed"]))
        reference = LayoutConditionedProposalRoleNet(seed=int(config["base_seed"]))
        state = torch.load(REPO_ROOT / P1_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"], strict=True)
        reference.load_state_dict(state["state_dict"], strict=True)
        reference.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for module in (model.base.proposal_head, model.base.proposal_residual):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        trainable_names = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
        if trainable_names != sorted(config["trainable_parameter_names"]):
            raise RuntimeError("OCR V15 P2 trainable parameter boundary changed")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        proposal_criterion = nn.CrossEntropyLoss()
        tensors = torch.from_numpy(values)
        proposal_targets = torch.from_numpy(proposal_labels)
        positive_indices = torch.nonzero(proposal_targets == 1, as_tuple=False).flatten()
        if len(positive_indices) != int(config["positive_proposal_count"]):
            raise RuntimeError("OCR V15 P2 positive training population changed")
        selection_stream = sha256()
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        for epoch in range(int(config["epochs"])):
            hard_negative = _hard_negative_indices(
                model, tensors, proposal_targets, int(config["hard_negative_count"]), int(config["batch_size"]),
            )
            selection_stream.update(hard_negative.numpy().astype(np.int64, copy=False).tobytes(order="C"))
            order = torch.cat((positive_indices, hard_negative))
            order = order.index_select(0, torch.randperm(len(order), generator=generator))
            model.train()
            cross_entropy_losses: list[float] = []
            rank_losses: list[float] = []
            total_losses: list[float] = []
            for start in range(0, len(order), int(config["batch_size"])):
                indices = order[start:start + int(config["batch_size"])]
                targets = proposal_targets.index_select(0, indices)
                logits = model(tensors.index_select(0, indices))[:, :2]
                cross_entropy = proposal_criterion(logits, targets)
                margin = logits[:, 1] - logits[:, 0]
                positive_margin = torch.sort(margin[targets == 1]).values
                negative_margin = torch.sort(margin[targets == 0], descending=True).values
                pair_count = min(len(positive_margin), len(negative_margin))
                if pair_count == 0:
                    raise RuntimeError("OCR V15 P2 ranking batch lost a proposal class")
                ranking = F.margin_ranking_loss(
                    positive_margin[:pair_count], negative_margin[:pair_count],
                    torch.ones(pair_count), margin=float(config["pairwise_margin"]),
                )
                loss = cross_entropy + float(config["pairwise_loss_weight"]) * ranking
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                optimizer_steps += 1
                cross_entropy_losses.append(float(cross_entropy.detach()))
                rank_losses.append(float(ranking.detach()))
                total_losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({
                    "epoch": epoch + 1,
                    "cross_entropy_loss": sum(cross_entropy_losses) / len(cross_entropy_losses),
                    "pairwise_rank_loss": sum(rank_losses) / len(rank_losses),
                    "total_loss": sum(total_losses) / len(total_losses),
                })
        if optimizer_steps != int(config["expected_optimizer_steps"]):
            raise RuntimeError("OCR V15 P2 optimizer-step count changed")

        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-layout-conditioned-proposal-role-v15-p2.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-layout-conditioned-proposal-role-v15-p2.onnx"
        parity_values = torch.from_numpy(values[:256])
        with torch.inference_mode():
            expected = model(parity_values).numpy()
            reference_roles = reference(parity_values)[:, 2:].numpy()
        role_invariance_error = float(np.max(np.abs(expected[:, 2:] - reference_roles)))
        if role_invariance_error != 0.0:
            raise RuntimeError("OCR V15 P2 changed frozen role logits")
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V15 P2 selection requires CPU execution only")
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
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-layout-conditioned-proposal-role-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v14_validation_fixture_bytes_scene_truth_or_case_identity_used": False,
            "p1_validation_case_detail_or_pixels_used_for_design": False,
            "p1_aggregate_metrics_only_used_for_design": True,
            "optimizer_steps": optimizer_steps, "weights_changed": True,
            "isolated_change": config["isolated_change"], "training_evidence": training_evidence,
            "trainable_parameter_names": trainable_names,
            "hard_negative_count": config["hard_negative_count"],
            "hard_negative_index_stream_sha256": selection_stream.hexdigest(),
            "pairwise_margin": config["pairwise_margin"],
            "pairwise_loss_weight": config["pairwise_loss_weight"], "loss_checkpoints": checkpoints,
            "p1_result_sha256": config["p1_result_sha256"],
            "p1_checkpoint_sha256": config["p1_checkpoint_sha256"],
            "role_invariance_maximum_absolute_error": role_invariance_error,
            "checkpoint_path": checkpoint_path.relative_to(REPO_ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint_path),
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
        "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"],
        "role_invariance_maximum_absolute_error": report["role_invariance_maximum_absolute_error"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
