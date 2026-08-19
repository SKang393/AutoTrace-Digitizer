# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use V31 P3 margin training after the aggregate P2 failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import complete_training_candidate
from ml.ocr.unanimous_structure_veto_v30 import train_p1 as trainer

from . import train_p1 as v31
from .dataset import load_archive, proposal_summary, split_fingerprint
from .model import RobustQuorumRecallProposalNet
from .pipeline import extract_relational_evidence
from .prepare_split import SOURCE_PATHS
from .protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    REVISION,
    SEED,
    TASK,
    THRESHOLDS,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/robust_quorum_recall_v31")
CANDIDATE_ID = "P3"
CONFIG_PATH = ROOT / "training/p3.json"
P2_RESULT_PATH = ROOT / "P2_RESULT.json"
SEAL_PATH = ROOT / "SPLIT_SEAL.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P3-run"
TRAIN_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-train.zip")
SELECTION_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-selection.zip")
PUBLIC_ARCHIVE_PATH = Path("artifacts/production-validation/ocr-v31-public.zip")
P2_CHECKPOINT_PATH = ROOT / "artifacts/P2-run/graph-text-robust-quorum-recall-v31-p2.pt"
P2_ONNX_PATH = ROOT / "artifacts/P2-run/graph-text-robust-quorum-recall-v31-p2.onnx"
P2_CHECKPOINT_SHA256 = "c2c5077c44538390bb71f73739c38aecdfa81b4c6d93799f01bc002d16a82e36"
P2_ONNX_SHA256 = "98ff06aef445cbdb0a9c7a7a376ee5b0eea51c691610ba0d4fe72203225b976a"
P2_RESULT_SHA256 = "34106e7a018be2964d733162b27292cef5db9bb448eaf3e999accbbd6065c4a3"
RUNNER_SOURCE_PATHS = (
    *SOURCE_PATHS,
    Path("ml/ocr/unanimous_structure_veto_v30/train_p1.py"),
    ROOT / "P1_RESULT.json",
    ROOT / "train_p2.py",
    P2_RESULT_PATH,
    ROOT / "train_p3.py",
)

_RAW_COMPLETE = complete_training_candidate


class MarginFineTunedRobustQuorumRecallProposalNet(RobustQuorumRecallProposalNet):
    """Load the complete P2 state before bounded V31 margin training."""

    def load_role_parent_state_dict(
        self, state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.load_state_dict(state_dict, strict=True)


def _single_candidate_acquisition_contract_satisfied(
    entry: dict[str, Any],
) -> bool:
    """Mirror the canonical acquire contract before reporting ready."""

    return (
        entry.get("execution_authorized") is True
        and entry.get("authorized_candidate_id") == CANDIDATE_ID
        and entry.get("preregistered_candidate_ids") == [CANDIDATE_ID]
        and CANDIDATE_ID not in entry.get("consumed_candidate_ids", [])
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _require_committed(path: Path) -> None:
    relative = path.as_posix()
    completed = subprocess.run(
        ("git", "show", f"HEAD:{relative}"), cwd=REPO_ROOT,
        check=False, capture_output=True,
    )
    local = REPO_ROOT / path
    if completed.returncode != 0 or not local.is_file() or completed.stdout != local.read_bytes():
        raise RuntimeError(f"OCR V31 P3 source must be committed unchanged: {relative}")


def _load_predecessor_state() -> dict[str, torch.Tensor]:
    payload = torch.load(
        REPO_ROOT / P2_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("OCR V31 P3 predecessor state is missing")
    return state


def _validate_stored_split(
    scenes: tuple[Any, ...], registered: dict[str, Any], name: str,
) -> None:
    summary = proposal_summary(scenes)
    if (
        split_fingerprint(scenes) != registered["split_fingerprint"]
        or any(summary[key] != registered["proposal_summary"][key] for key in summary)
    ):
        raise RuntimeError(f"OCR V31 P3 {name} stored fixtures violate the seal")


def _p2_is_terminal(result: dict[str, Any]) -> bool:
    metrics = result.get("selected_threshold_metrics", {})
    comparisons = result.get("threshold_comparisons", [])
    return bool(
        result.get("schema") == "graphreader.ocr-robust-quorum-recall-result.v1"
        and result.get("candidate_id") == "P2"
        and result.get("status") == "failed_selection"
        and result.get("optimizer_steps") == 0
        and result.get("weights_changed") is False
        and result.get("selection_archive_read_count") == 1
        and result.get("passing_threshold_window") == []
        and result.get("selected_threshold") == 0.75
        and metrics.get("scene_count", result.get("scene_count")) == 192
        and metrics.get("exact_scene_count") == 192
        and metrics.get("true_positives") == 1536
        and metrics.get("false_positives") == 0
        and metrics.get("false_negatives") == 0
        and metrics.get("duplicate_region_count") == 0
        and metrics.get("prohibited_structure_hits") == 0
        and len(comparisons) == 5
        and all(item.get("false_positives") == 1 for item in comparisons[:4])
        and all(item.get("prohibited_structure_hits") == 1 for item in comparisons[:4])
        and comparisons[4].get("false_positives") == 0
        and result.get("case_detail_or_pixels_inspected") is False
        and result.get("public_gate_archive_opened") is False
        and "cases" not in result
        and "predictions" not in result
        and "truths" not in result
    )


def preflight(*, require_authorized: bool = True) -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIG_PATH)
    expected = {
        "schema": "graphreader.ocr-robust-quorum-recall-candidate.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "candidate_limit": 3,
        "architecture": "two-of-three-robust-route-quorum-margin-finetune-v2",
        "objective": "robust-quorum-asymmetric-margin-finetune-v1",
        "model_license": "Apache-2.0",
        "seed": SEED + 2,
        "learning_rate": 0.0001,
        "weight_decay": 0.0005,
        "epochs": 4,
        "scene_batch_size": 1,
        "expected_optimizer_steps": 1536,
        "gradient_clip_norm": 5.0,
        "unanimous_objective_weight": 1.0,
        "per_route_objective_weight": 0.2,
        "worst_route_objective_weight": 0.2,
        "route_diversity_weight": 0.05,
        "proposal_cross_entropy_weight": 1.0,
        "false_positive_weight": 12.0,
        "positive_logit_margin_floor": 2.5,
        "negative_logit_margin_ceiling": -4.0,
        "positive_floor_weight": 2.0,
        "negative_ceiling_weight": 12.0,
        "scene_separation_logit_margin_minimum": 6.5,
        "scene_separation_weight": 5.0,
        "hard_negative_top_k": 8,
        "hard_negative_weight": 8.0,
        "proposal_selection": "all_frozen_production_proposals_no_detector_prefilter",
        "complete_proposal_negative_cap_per_scene": 10000,
        "detector_prefilter_applied": False,
        "recognition_batch_size": 64,
        "runtime_numeric_precision": "float32",
        "candidate_onnx_graph_optimization_level": "ORT_DISABLE_ALL",
        "predecessor_checkpoint_path": P2_CHECKPOINT_PATH.as_posix(),
        "predecessor_checkpoint_sha256": P2_CHECKPOINT_SHA256,
        "predecessor_onnx_path": P2_ONNX_PATH.as_posix(),
        "predecessor_onnx_sha256": P2_ONNX_SHA256,
        "p2_result_path": P2_RESULT_PATH.as_posix(),
        "p2_result_sha256": P2_RESULT_SHA256,
        "selection_thresholds": list(THRESHOLDS),
        "minimum_consecutive_passing_thresholds": 3,
        "onnx_parity_maximum_absolute_error": ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR,
        "validation_or_public_pixels_used_for_training": False,
        "case_level_predecessor_evidence_used": False,
        "selection_evaluation_limit": 1,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "private_or_article_images": False,
        "chandler_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"OCR V31 P3 config field mismatch: {key}")
    if config.get("expected_runner_source_bundle_sha256") != source_bundle_sha256(
        REPO_ROOT, RUNNER_SOURCE_PATHS,
    ):
        raise RuntimeError("OCR V31 P3 runner source bundle changed")
    for path in (*RUNNER_SOURCE_PATHS, CONFIG_PATH):
        _require_committed(path)
    if sha256_file(REPO_ROOT / SEAL_PATH) != config.get("split_seal_sha256"):
        raise RuntimeError("OCR V31 split seal changed before P3")
    seal = _read_json(REPO_ROOT / SEAL_PATH)
    for key, value in {
        "schema": "graphreader.ocr-robust-quorum-recall-split-seal.v1",
        "revision": REVISION,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "candidate_execution_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "private_data": False,
        "chandler_used": False,
        "production_approval": False,
        "release_eligible": False,
    }.items():
        if seal.get(key) != value:
            raise RuntimeError(f"OCR V31 split seal field changed before P3: {key}")
    head = v31._repository_head()
    if not v31._is_ancestor(str(seal.get("source_commit", "")), head):
        raise RuntimeError("OCR V31 split source commit is not an ancestor")
    for split, (path, config_key) in {
        "train": (TRAIN_ARCHIVE_PATH, "train_fixture_archive_sha256"),
        "validation": (SELECTION_ARCHIVE_PATH, "selection_fixture_archive_sha256"),
        "sealed_public": (PUBLIC_ARCHIVE_PATH, "public_fixture_archive_sha256"),
    }.items():
        actual = sha256_file(REPO_ROOT / path)
        if actual != seal["splits"][split]["archive_sha256"] or actual != config.get(config_key):
            raise RuntimeError(f"OCR V31 {split} archive changed before P3")
    for relative, expected_hash in {
        DETECTOR_PATH: DETECTOR_SHA256,
        RECOGNIZER_PATH: RECOGNIZER_SHA256,
        RECOGNIZER_YAML_PATH: RECOGNIZER_YAML_SHA256,
        TRIGGER_RESULT_PATH: TRIGGER_RESULT_SHA256,
        P2_CHECKPOINT_PATH.as_posix(): P2_CHECKPOINT_SHA256,
        P2_ONNX_PATH.as_posix(): P2_ONNX_SHA256,
        P2_RESULT_PATH.as_posix(): P2_RESULT_SHA256,
    }.items():
        path = REPO_ROOT / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"OCR V31 P3 exact input changed: {relative}")
    if not _p2_is_terminal(_read_json(REPO_ROOT / P2_RESULT_PATH)):
        raise RuntimeError("OCR V31 aggregate-only P2 trigger changed")
    if (REPO_ROOT / CANONICAL_OUTPUT).exists():
        raise RuntimeError("OCR V31 P3 output already exists")
    ledger = _read_json(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next((item for item in ledger.get("revisions", []) if item.get("task") == TASK and item.get("revision") == REVISION), None)
    if (
        entry is None
        or entry.get("status") != "candidate_3_preregistered"
        or entry.get("consumed_candidate_ids") != ["P1", "P2"]
        or entry.get("remaining_unregistered_candidate_ids") != []
        or entry.get("selection_evaluations") != 2
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("public_gate_evaluations") != 0
    ):
        raise RuntimeError("OCR V31 P3 ledger state is not preregistered")
    if not _single_candidate_acquisition_contract_satisfied(entry):
        raise RuntimeError("OCR V31 P3 ledger cannot satisfy the training acquisition contract")
    if require_authorized and (
        entry.get("execution_authorized") is not True
        or entry.get("authorized_candidate_id") != CANDIDATE_ID
        or config.get("candidate_execution_authorized") is not True
    ):
        raise RuntimeError("OCR V31 P3 execution is not separately authorized")
    if not require_authorized and config.get("candidate_execution_authorized") not in (False, True):
        raise RuntimeError("OCR V31 P3 authorization field is invalid")
    return {"config": config, "seal": seal, "entry": entry, "head": head}


def _complete_with_p3_payload_names(
    authorization, *, status: str, report_sha256: str,
):
    report_path = REPO_ROOT / CANONICAL_OUTPUT / "candidate-report.json"
    report = _read_json(report_path)
    report["schema"] = "graphreader.ocr-robust-quorum-recall-candidate-report.v1"
    report["predecessor_checkpoint_reused"] = True
    report["predecessor_checkpoint_path"] = P2_CHECKPOINT_PATH.as_posix()
    report["predecessor_checkpoint_sha256"] = P2_CHECKPOINT_SHA256
    report["initial_state_loaded_strict"] = True
    report["aggregate_design_basis"] = (
        "P2 aggregate threshold counts, zero-error selected-threshold metrics, "
        "parity, optimizer count, and closed-public state only"
    )
    if status in {"selected", "failed_selection"}:
        old_checkpoint = REPO_ROOT / CANONICAL_OUTPUT / "graph-text-unanimous-structure-veto-v30-p1.pt"
        old_onnx = REPO_ROOT / CANONICAL_OUTPUT / "graph-text-unanimous-structure-veto-v30-p1.onnx"
        new_checkpoint = old_checkpoint.with_name("graph-text-robust-quorum-recall-v31-p3.pt")
        new_onnx = old_onnx.with_name("graph-text-robust-quorum-recall-v31-p3.onnx")
        if not old_checkpoint.is_file() or not old_onnx.is_file():
            raise RuntimeError("OCR V31 P3 payload rename source is incomplete")
        if new_checkpoint.exists() or new_onnx.exists():
            raise RuntimeError("OCR V31 P3 payload rename target already exists")
        old_checkpoint.rename(new_checkpoint)
        old_onnx.rename(new_onnx)
        report["checkpoint_path"] = new_checkpoint.relative_to(REPO_ROOT).as_posix()
        report["onnx_path"] = new_onnx.relative_to(REPO_ROOT).as_posix()
    report_path.write_bytes(canonical_json_bytes(report))
    return _RAW_COMPLETE(
        authorization,
        status=status,
        report_sha256=sha256_file(report_path),
    )


def train_candidate(output_dir: Path) -> dict[str, object]:
    originals = {
        "ROOT": trainer.ROOT,
        "CANDIDATE_ID": trainer.CANDIDATE_ID,
        "CONFIG_PATH": trainer.CONFIG_PATH,
        "SEAL_PATH": trainer.SEAL_PATH,
        "CANONICAL_OUTPUT": trainer.CANONICAL_OUTPUT,
        "TRAIN_ARCHIVE_PATH": trainer.TRAIN_ARCHIVE_PATH,
        "SELECTION_ARCHIVE_PATH": trainer.SELECTION_ARCHIVE_PATH,
        "PUBLIC_ARCHIVE_PATH": trainer.PUBLIC_ARCHIVE_PATH,
        "RUNNER_SOURCE_PATHS": trainer.RUNNER_SOURCE_PATHS,
        "REVISION": trainer.REVISION,
        "SEED": trainer.SEED,
        "THRESHOLDS": trainer.THRESHOLDS,
        "UnanimousStructureVetoProposalNet": trainer.UnanimousStructureVetoProposalNet,
        "load_archive": trainer.load_archive,
        "proposal_summary": trainer.proposal_summary,
        "split_fingerprint": trainer.split_fingerprint,
        "extract_relational_evidence": trainer.extract_relational_evidence,
        "_load_role_parent_state": trainer._load_role_parent_state,
        "_validate_stored_split": trainer._validate_stored_split,
        "preflight": trainer.preflight,
        "complete_training_candidate": trainer.complete_training_candidate,
    }
    trainer.ROOT = ROOT
    trainer.CANDIDATE_ID = CANDIDATE_ID
    trainer.CONFIG_PATH = CONFIG_PATH
    trainer.SEAL_PATH = SEAL_PATH
    trainer.CANONICAL_OUTPUT = CANONICAL_OUTPUT
    trainer.TRAIN_ARCHIVE_PATH = TRAIN_ARCHIVE_PATH
    trainer.SELECTION_ARCHIVE_PATH = SELECTION_ARCHIVE_PATH
    trainer.PUBLIC_ARCHIVE_PATH = PUBLIC_ARCHIVE_PATH
    trainer.RUNNER_SOURCE_PATHS = RUNNER_SOURCE_PATHS
    trainer.REVISION = REVISION
    trainer.SEED = SEED + 2
    trainer.THRESHOLDS = THRESHOLDS
    trainer.UnanimousStructureVetoProposalNet = MarginFineTunedRobustQuorumRecallProposalNet
    trainer.load_archive = load_archive
    trainer.proposal_summary = proposal_summary
    trainer.split_fingerprint = split_fingerprint
    trainer.extract_relational_evidence = extract_relational_evidence
    trainer._load_role_parent_state = _load_predecessor_state
    trainer._validate_stored_split = _validate_stored_split
    trainer.preflight = lambda: preflight(require_authorized=True)
    trainer.complete_training_candidate = _complete_with_p3_payload_names
    try:
        trainer.train_candidate(output_dir)
        return _read_json(output_dir / "candidate-report.json")
    finally:
        for name, value in originals.items():
            setattr(trainer, name, value)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight:
        evidence = preflight(require_authorized=True)
        print(json.dumps({
            "head": evidence["head"],
            "runner_source_bundle_sha256": source_bundle_sha256(
                REPO_ROOT, RUNNER_SOURCE_PATHS,
            ),
            "ready": True,
        }, sort_keys=True))
        return 0
    report = train_candidate(REPO_ROOT / CANONICAL_OUTPUT)
    print(json.dumps({
        "status": report["status"],
        "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report.get("selected_threshold"),
        "passing_threshold_window": report.get("passing_threshold_window", []),
        "onnx_parity_maximum_absolute_error": report.get(
            "onnx_parity_maximum_absolute_error"
        ),
        "selection_metrics": report.get("selection_metrics"),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANONICAL_OUTPUT", "CONFIG_PATH", "RUNNER_SOURCE_PATHS",
    "MarginFineTunedRobustQuorumRecallProposalNet", "preflight", "train_candidate",
]
