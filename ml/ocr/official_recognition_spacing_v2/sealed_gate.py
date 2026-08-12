# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use truth-hidden public gate for the selected spacing repair."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ml.markers.gate_seal import (
    acquire_gate_seal, canonical_json_bytes, complete_gate_seal, require_committed_sources,
    sha256_file,
)
from ml.markers.training_budget import CANONICAL_LEDGER_PATH
from ml.ocr.official_bakeoff.production_evaluate import _cpu_session, read_character_alphabet
from .evaluate import (
    INFERENCE_YAML_PATH, MODEL_PATH, PUBLIC_SEAL_PATH,
    _load, _load_partition, evaluate_partition,
)
from .protocol import CANDIDATE_ID, GATES, MODEL_SHA256, PUBLIC_GATE_CONFIG, REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/official_recognition_spacing_v2")
RESULT_PATH = ROOT / "P2_RESULT.json"
AUTHORIZED_PUBLIC_GATE_CONFIG_PATH = ROOT / "gates/sealed-public-p2-authorized.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "prepare_split.py", ROOT / "spacing.py", ROOT / "protocol.py", ROOT / "evaluate.py",
    ROOT / "sealed_gate.py", Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
)


def evaluate_public(*, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"Recognition spacing public output exists: {output_path}")
    require_committed_sources(REPO_ROOT, (*EVALUATOR_SOURCE_PATHS, CANONICAL_LEDGER_PATH, RESULT_PATH))
    result = _load(REPO_ROOT / RESULT_PATH)
    ledger = _load(REPO_ROOT / CANONICAL_LEDGER_PATH)
    entry = next(
        item for item in ledger["revisions"]
        if item.get("task") == TASK and item.get("revision") == REVISION
    )
    selection_report_path = REPO_ROOT / str(result["selection_report_path"])
    selection_report_sha256 = sha256_file(selection_report_path)
    spacing_source_sha256 = sha256_file(REPO_ROOT / ROOT / "spacing.py")
    if (
        result.get("status") != "selected_public_gate_pending"
        or result.get("candidate_id") != CANDIDATE_ID
        or result.get("model_onnx_sha256") != MODEL_SHA256
        or result.get("selection_report_sha256") != selection_report_sha256
        or result.get("spacing_source_sha256") != spacing_source_sha256
        or result.get("public_gate_evaluations") != 0
        or result.get("public_gate_archive_opened") is not False
        or entry.get("status") != "candidate_2_selected_public_gate_pending"
        or entry.get("consumed_candidate_ids") != ["P1", CANDIDATE_ID]
        or entry.get("execution_authorized") is not False
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_authorized_candidate_id") != CANDIDATE_ID
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_archive_opened") is not False
        or entry.get("p2_result_sha256") != sha256_file(REPO_ROOT / RESULT_PATH)
        or entry.get("p2_selection_report_sha256") != selection_report_sha256
    ):
        raise RuntimeError("Recognition spacing public gate is not authorized by exact selection evidence")
    if sha256_file(REPO_ROOT / MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("Exact official recognition ONNX changed")
    manifest, archive = _load_partition(PUBLIC_SEAL_PATH, "sealed_public")
    seal = _load(REPO_ROOT / PUBLIC_SEAL_PATH)
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT,
        task=TASK,
        revision=f"{REVISION}-public-v1",
        candidate_hashes={
            "onnx_sha256": MODEL_SHA256,
            "spacing_source_sha256": spacing_source_sha256,
            "selection_report_sha256": selection_report_sha256,
        },
        dataset_manifest_sha256=str(seal["private_manifest_sha256"]),
        split_config_path=AUTHORIZED_PUBLIC_GATE_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=PUBLIC_GATE_CONFIG,
    )
    evaluated = evaluate_partition(
        manifest,
        archive,
        _cpu_session(REPO_ROOT / MODEL_PATH),
        read_character_alphabet(REPO_ROOT / INFERENCE_YAML_PATH),
        candidate_id=CANDIDATE_ID,
    )
    metrics = evaluated["metrics"]
    passed = bool(metrics["passed"] and metrics["inference_calls"] == len(manifest["cases"]))
    report: dict[str, object] = {
        "schema": "graphreader.ocr-official-recognition-spacing-public-gate.v1",
        "task": TASK,
        "revision": f"{REVISION}-public-v1",
        "candidate_id": CANDIDATE_ID,
        "status": "pass" if passed else "fail",
        "evaluation_count": 1,
        "model_onnx_sha256": MODEL_SHA256,
        "spacing_source_sha256": spacing_source_sha256,
        "selection_report_sha256": selection_report_sha256,
        "fixture_archive_sha256": sha256(archive).hexdigest(),
        "private_manifest_sha256": seal["private_manifest_sha256"],
        "provider": "CPUExecutionProvider",
        "gate_requirements": GATES,
        **evaluated,
        "seal_binding": gate.binding,
        "canonical_seal_key": gate.key,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    complete_gate_seal(gate, status=report["status"], report_sha256=sha256_file(output_path))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public(output_path=REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
