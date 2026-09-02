# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Evaluate base and repaired raw proposals on synthetic V32 dev data."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.component_context_detector_v7.dataset import proposals

from .dataset import build_split, repaired_proposals
from .pipeline import score_proposals
from .protocol import (
    DATASET_MODULE,
    PRECISION_MINIMUM,
    PROPOSAL_SOURCE,
    RAW_PROPOSAL_DIAGNOSTIC_PATH,
    RAW_PROPOSAL_DIAGNOSTIC_SHA256,
    RECALL_MINIMUM,
    REVISION,
    V33_RESULT_PATH,
    V33_RESULT_SHA256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/real_range_proposal_v34")
CONFIG_PATH = ROOT / "evaluation/p1.json"
EVIDENCE_POLICY_PATH = Path("ml/policy/evidence-policy.json")
SOURCE_PATHS = (
    ROOT / "dataset.py",
    ROOT / "pipeline.py",
    ROOT / "protocol.py",
    ROOT / "evaluate.py",
    Path("ml/ocr/real_range_classifier_finetune_v32/dataset.py"),
    Path("ml/ocr/real_range_classifier_finetune_v32/protocol.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
)


def run() -> dict[str, object]:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    actual_source_hash = source_bundle_sha256(REPO_ROOT, SOURCE_PATHS)
    if config["expected_runner_source_bundle_sha256"] != actual_source_hash:
        raise RuntimeError("V34 source bundle hash differs from the reviewed diagnostic config")
    if sha256_file(REPO_ROOT / RAW_PROPOSAL_DIAGNOSTIC_PATH) != RAW_PROPOSAL_DIAGNOSTIC_SHA256:
        raise RuntimeError("V34 raw-proposal trigger diagnostic changed")
    if sha256_file(REPO_ROOT / V33_RESULT_PATH) != V33_RESULT_SHA256:
        raise RuntimeError("V34 V33 trigger result changed")
    scenes = build_split("dev")
    started = time.perf_counter()
    base = score_proposals(scenes, proposals)
    repaired = score_proposals(scenes, repaired_proposals)
    gates = {
        "raw_proposal_precision": repaired["precision"] >= PRECISION_MINIMUM,
        "raw_proposal_recall": repaired["recall"] >= RECALL_MINIMUM,
    }
    return {
        "schema": "graphreader.ocr-real-range-proposal-v34-result.v1",
        "revision": REVISION,
        "status": "pass" if all(gates.values()) else "failed_dev_diagnostic",
        "evaluation_kind": "non_candidate_diagnostic",
        "synthetic_only": True,
        "private_or_article_images": False,
        "public_or_sealed_reads": 0,
        "data_module": DATASET_MODULE.as_posix(),
        "proposal_source": PROPOSAL_SOURCE.as_posix(),
        "source_bundle_paths": [path.as_posix() for path in SOURCE_PATHS],
        "source_bundle_sha256": actual_source_hash,
        "raw_proposal_diagnostic_path": RAW_PROPOSAL_DIAGNOSTIC_PATH,
        "raw_proposal_diagnostic_sha256": RAW_PROPOSAL_DIAGNOSTIC_SHA256,
        "v33_result_path": V33_RESULT_PATH,
        "v33_result_sha256": V33_RESULT_SHA256,
        "evidence_policy_path": EVIDENCE_POLICY_PATH.as_posix(),
        "evidence_policy_sha256": sha256_file(REPO_ROOT / EVIDENCE_POLICY_PATH),
        "strategies": {"v32_base": base, "v34_deterministic_expansion": repaired},
        "gates": gates,
        "training_use": False,
        "production_approval": False,
        "release_eligible": False,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    if args.output is not None:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
