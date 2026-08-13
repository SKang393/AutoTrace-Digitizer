# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for OCR V13."""

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
from .dataset import build_split, proposal_summary, split_fingerprint, training_examples
from .model import MorphologyProposalRoleNet
from .pipeline import evaluate_thresholds
from .protocol import (
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION, ROLE_ACCURACY_MINIMUM,
    ROLE_CLASS_ACCURACY_MINIMUM, TASK,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/morphology_proposal_role_v13")
CANDIDATE_ID = "P1"
CONFIG_PATH = ROOT / "training/p1.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "train_p1.py",
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


def _balanced_order(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    negative = torch.nonzero(labels == 0, as_tuple=False).flatten()
    positive = torch.nonzero(labels == 1, as_tuple=False).flatten()
    if len(negative) == 0 or len(positive) == 0:
        raise RuntimeError("OCR V13 balanced sampler requires both proposal classes")
    target = max(len(negative), len(positive))

    def expand(values: torch.Tensor) -> torch.Tensor:
        shuffled = values.index_select(0, torch.randperm(len(values), generator=generator))
        return shuffled.repeat((target + len(values) - 1) // len(values))[:target]

    paired = torch.stack((expand(negative), expand(positive)), dim=1).flatten()
    return paired.index_select(0, torch.randperm(len(paired), generator=generator))


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model, example, path, input_names=["region_proposals"], output_names=["proposal_role_logits"],
        dynamic_axes={"region_proposals": {0: "proposal_count"}, "proposal_role_logits": {0: "proposal_count"}},
        opset_version=18, dynamo=False,
    )


def train_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V13 output exists: {output_dir}")
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
            raise RuntimeError("OCR V13 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V13 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V13 public seal changed")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        train, validation = build_split("train"), build_split("validation")
        if split_fingerprint(train) != selection["train"]["split_fingerprint"]:
            raise RuntimeError("OCR V13 train split changed")
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V13 validation split changed")
        if proposal_summary(train) != {key: selection["train"][key] for key in proposal_summary(train)}:
            raise RuntimeError("OCR V13 train proposals changed")
        values, proposal_labels, role_labels, training_evidence = training_examples(
            train, negative_cap_per_scene=int(config["negative_cap_per_scene"]),
        )
        for key in (
            "scene_count", "negative_cap_per_scene", "negative_sampling", "negative_family_counts",
            "proposal_count", "positive_proposal_count", "negative_proposal_count", "tensor_label_stream_sha256",
        ):
            if training_evidence[key] != config[key]:
                raise RuntimeError(f"OCR V13 training evidence changed: {key}")
        if any(training_evidence[key] is not False for key in (
            "validation_or_public_pixels_used", "predecessor_fixture_bytes_used",
            "v12_public_fixture_bytes_scene_truth_or_case_identity_used",
        )):
            raise RuntimeError("OCR V13 training data violated preregistered scope")
        generator = _configure(int(config["seed"]))
        model = MorphologyProposalRoleNet(seed=int(config["seed"]))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]),
        )
        proposal_criterion = nn.CrossEntropyLoss()
        role_counts = np.bincount(role_labels[role_labels >= 0], minlength=8).astype(np.float32)
        role_weights = role_counts.sum() / np.maximum(role_counts, 1.0)
        role_weights /= role_weights.mean()
        role_criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(role_weights))
        tensors = torch.from_numpy(values)
        proposal_targets = torch.from_numpy(proposal_labels)
        role_targets = torch.from_numpy(role_labels)
        checkpoints: list[dict[str, float | int]] = []
        phase = "training"
        model.train()
        for epoch in range(int(config["epochs"])):
            order = _balanced_order(proposal_targets, generator)
            losses: list[float] = []
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
                losses.append(float(loss.detach()))
            if epoch in {0, int(config["epochs"]) // 2, int(config["epochs"]) - 1}:
                checkpoints.append({"epoch": epoch + 1, "multitask_loss": sum(losses) / len(losses)})

        phase = "export"
        model.eval()
        checkpoint_path = output_dir / "graph-text-morphology-proposal-role-v13-p1.pt"
        torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        onnx_path = output_dir / "graph-text-morphology-proposal-role-v13-p1.onnx"
        parity_values = torch.from_numpy(values[:256])
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V13 selection requires CPU execution only")
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        calls = 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            return np.asarray(session.run(None, {"region_proposals": np.ascontiguousarray(input_values)})[0], dtype=np.float32)

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
            "schema": "graphreader.ocr-morphology-proposal-role-candidate.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False,
            "v12_public_fixture_bytes_scene_truth_or_case_identity_used": False,
            "optimizer_steps": optimizer_steps, "training_evidence": training_evidence,
            "loss_checkpoints": checkpoints,
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
            "schema": "graphreader.ocr-morphology-proposal-role-failure.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner",
            "phase": phase, "optimizer_steps": optimizer_steps, "exception_type": type(error).__name__,
            "exception_message": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
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
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
