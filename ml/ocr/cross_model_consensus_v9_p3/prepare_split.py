# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize both P3 fixture identities before any model execution."""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import build_split, save_archive
from .protocol import CANDIDATE_ID, P2_PUBLIC_RESULT_SHA256, V11_PUBLIC_RESULT_SHA256


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    Path("ml/ocr/cross_model_consensus_v9_p3/PROTOCOL.json"),
    Path("ml/ocr/cross_model_consensus_v9_p3/dataset.py"),
    Path("ml/ocr/cross_model_consensus_v9_p3/prepare_split.py"),
    Path("ml/ocr/cross_model_consensus_v9_p3/protocol.py"),
    Path("ml/ocr/selected_confidence_public_gate_v9_p2/PUBLIC_GATE_RESULT.json"),
    Path("ml/ocr/composite_proposal_role_v11/PUBLIC_GATE_RESULT.json"),
    Path("src/GraphReader.Ocr/OcrV9P3CrossModelConsensusPipeline.cs"),
    Path("tests/GraphReader.Ocr.Tests/OcrV9P3DirectCorpusTests.cs"),
)


def materialize(selection_output: Path, public_output: Path, seal_output: Path) -> dict[str, object]:
    for path in (selection_output, public_output, seal_output):
        if path.exists():
            raise RuntimeError(f"Refusing to replace frozen P3 identity: {path}")
    selection_secret = secrets.randbits(128)
    public_secret = secrets.randbits(128)
    if selection_secret == public_secret:
        raise RuntimeError("P3 split secrets unexpectedly matched")
    selection = save_archive(build_split("selection", selection_secret), "selection", selection_output)
    public = save_archive(build_split("sealed_public", public_secret), "sealed_public", public_output)
    source_sha256 = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in SOURCE_PATHS
    }
    if source_sha256[
        "ml/ocr/selected_confidence_public_gate_v9_p2/PUBLIC_GATE_RESULT.json"
    ] != P2_PUBLIC_RESULT_SHA256:
        raise RuntimeError("P2 aggregate predecessor result changed")
    if source_sha256[
        "ml/ocr/composite_proposal_role_v11/PUBLIC_GATE_RESULT.json"
    ] != V11_PUBLIC_RESULT_SHA256:
        raise RuntimeError("V11 public predecessor result changed")
    seal = {
        "schema": "graphreader.ocr-cross-model-consensus-split-seal.v1",
        "candidate_id": CANDIDATE_ID,
        "selection_archive_path": selection_output.relative_to(REPO_ROOT).as_posix(),
        "selection_archive_sha256": selection["fixture_archive_sha256"],
        "selection_manifest_sha256": selection["fixture_manifest_sha256"],
        "selection_scene_count": selection["scene_count"],
        "selection_truth_region_count": selection["truth_region_count"],
        "public_archive_path": public_output.relative_to(REPO_ROOT).as_posix(),
        "public_archive_sha256": public["fixture_archive_sha256"],
        "public_manifest_sha256": public["fixture_manifest_sha256"],
        "public_scene_count": public["scene_count"],
        "public_truth_region_count": public["truth_region_count"],
        "source_sha256": source_sha256,
        "secret_seeds_serialized": False,
        "model_execution_count_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "selection_execution_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
    }
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    seal_output.write_bytes(canonical_json_bytes(seal))
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    seal = materialize(
        (REPO_ROOT / args.selection_output).resolve(),
        (REPO_ROOT / args.public_output).resolve(),
        (REPO_ROOT / args.seal_output).resolve(),
    )
    print(f"selection_archive_sha256={seal['selection_archive_sha256']}")
    print(f"public_archive_sha256={seal['public_archive_sha256']}")
    print(f"selection_manifest_sha256={seal['selection_manifest_sha256']}")
    print(f"public_manifest_sha256={seal['public_manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
