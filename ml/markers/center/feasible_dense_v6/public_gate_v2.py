# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Corrected source-bound one-use public wrapper for feasible dense V6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.markers.center.feasible_dense_v6.public_gate import GATE_CONFIGURATION, _run_opened_gate
from ml.markers.gate_seal import acquire_gate_seal, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/markers/center/feasible_dense_v6"
TASK = "marker-center"
REVISION = "marker-center-feasible-dense-v6-public-v2"
GATE_PATH = Path("ml/markers/center/feasible_dense_v6/gates/sealed-public-v2.json")
EVALUATOR_SOURCE_PATHS = (
    Path("ml/markers/center/artifact_mask_public_gate.py"),
    Path("ml/markers/center/dense_contract_v5/dataset.py"),
    Path("ml/markers/center/dense_contract_v5/train_p1.py"),
    Path("ml/markers/center/feasible_dense_v6/dataset.py"),
    Path("ml/markers/center/feasible_dense_v6/public_gate.py"),
    Path("ml/markers/center/feasible_dense_v6/public_gate_v2.py"),
)


def run(candidate_report_path: Path, output_path: Path) -> dict[str, object]:
    gate = json.loads((REPO_ROOT / GATE_PATH).read_text(encoding="utf-8"))
    candidate = json.loads(candidate_report_path.read_text(encoding="utf-8"))
    if candidate.get("selection_gate_passed") is not True:
        raise RuntimeError("Public V2 gate requires a passing visible-selection candidate")
    ledger = json.loads(
        (REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text(encoding="utf-8")
    )
    entry = next(item for item in ledger["revisions"] if item.get("revision") == "marker-center-feasible-dense-v6")
    candidate_id = str(candidate.get("candidate_id"))
    if (
        entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_revision") != REVISION
        or entry.get("public_gate_authorized_candidate_id") != candidate_id
    ):
        raise RuntimeError("Feasible dense V6 public V2 gate is not separately authorized")
    candidate_report_sha256 = sha256_file(candidate_report_path)
    if entry.get("public_gate_authorized_candidate_report_sha256") != candidate_report_sha256:
        raise RuntimeError("Authorized V2 candidate report identity changed")
    onnx_path = REPO_ROOT / str(candidate["onnx_path"])
    onnx_sha256 = sha256_file(onnx_path)
    if onnx_sha256 != candidate["onnx_sha256"] or entry.get("public_gate_authorized_onnx_sha256") != onnx_sha256:
        raise RuntimeError("Authorized V2 candidate ONNX identity changed")
    dataset_path = ROOT / "PUBLIC_DATASET_MANIFEST.json"
    split_seal_path = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
    split_seal = json.loads(split_seal_path.read_text(encoding="utf-8"))
    archive_path = REPO_ROOT / str(split_seal["fixture_archive_path"])
    if sha256_file(archive_path) != gate["expected_public_fixture_archive_sha256"]:
        raise RuntimeError("Truth-hidden feasible dense V6 public archive changed")
    seal = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_hashes={
            "candidate_report_sha256": candidate_report_sha256,
            "onnx_sha256": onnx_sha256,
        },
        dataset_manifest_sha256=sha256_file(dataset_path),
        split_config_path=GATE_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=GATE_CONFIGURATION,
    )
    return _run_opened_gate(
        candidate=candidate,
        candidate_report_path=candidate_report_path,
        onnx_path=onnx_path,
        archive_path=archive_path,
        dataset_path=dataset_path,
        split_seal_path=split_seal_path,
        seal=seal,
        output_path=output_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.candidate_report.resolve(), arguments.output.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
