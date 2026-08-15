# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize one truth-hidden public identity without executing models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import secrets

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import build_public_split, save_public_archive
from .protocol import CANDIDATE_ID, P2_SELECTION_RESULT_SHA256, configuration


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = Path(__file__).resolve().parent
SOURCE_PATHS = (
    Path("ml/ocr/selected_confidence_public_gate_v9_p2/PROTOCOL.json"),
    Path("ml/ocr/selected_confidence_public_gate_v9_p2/dataset.py"),
    Path("ml/ocr/selected_confidence_public_gate_v9_p2/prepare_split.py"),
    Path("ml/ocr/selected_confidence_public_gate_v9_p2/protocol.py"),
    Path("ml/ocr/selected_confidence_acceptance_v9_p2/P2_SELECTION_RESULT.json"),
    Path("src/GraphReader.Ocr/OcrV9P2CandidateCompositionPipeline.cs"),
    Path("tests/GraphReader.Ocr.Tests/OcrV9P2DirectPublicCorpusTests.cs"),
)


def _sha256_bytes(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def prepare(output: Path, seal_output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to replace public archive: {output}")
    if seal_output.exists():
        raise FileExistsError(f"Refusing to replace public seal: {seal_output}")
    if json.loads((MODULE_ROOT / "PROTOCOL.json").read_text(encoding="utf-8")) != configuration():
        raise RuntimeError("Tracked public protocol differs from the Python contract")
    selection_result = REPO_ROOT / "ml/ocr/selected_confidence_acceptance_v9_p2/P2_SELECTION_RESULT.json"
    if sha256_file(selection_result) != P2_SELECTION_RESULT_SHA256:
        raise RuntimeError("P2 selection result checksum changed")
    result = json.loads(selection_result.read_text(encoding="utf-8"))
    if result.get("selection_gates_passed") is not True or result.get("execution_consumed") is not True:
        raise RuntimeError("P2 did not pass and consume its frozen selection")

    secret_seed = secrets.randbits(128)
    summary = save_public_archive(build_public_split(secret_seed), output)
    source_sha256 = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in SOURCE_PATHS
    }
    seal = {
        "schema": "graphreader.ocr-selected-confidence-public-seal.v1",
        "candidate_id": CANDIDATE_ID,
        "fixture_archive_path": output.relative_to(REPO_ROOT).as_posix(),
        **summary,
        "source_sha256": source_sha256,
        "p2_selection_result_sha256": P2_SELECTION_RESULT_SHA256,
        "model_execution_count_at_freeze": 0,
        "public_execution_authorized": False,
        "public_gate_evaluations": 0,
        "marker_creation_evaluated": False,
        "production_approval": False,
        "release_eligible": False,
        "secret_seed_serialized": False,
    }
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    seal_bytes = canonical_json_bytes(seal)
    seal_output.write_bytes(seal_bytes)
    return seal | {"public_seal_sha256": _sha256_bytes(seal_bytes)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.output.resolve(), args.seal_output.resolve())
    print(f"fixture_archive_sha256={result['fixture_archive_sha256']}")
    print(f"fixture_manifest_sha256={result['fixture_manifest_sha256']}")
    print(f"public_seal_sha256={result['public_seal_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
