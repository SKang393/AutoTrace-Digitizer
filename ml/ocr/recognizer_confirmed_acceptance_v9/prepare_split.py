# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Materialize the V9 visible-selection identity once without model execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import secrets

import numpy as np
import PIL
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import build_selection_split, save_selection_archive
from .protocol import configuration


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    Path("ml/markers/gate_seal.py"),
    Path("ml/ocr/production_csharp_marker_gate_v4/dataset.py"),
    Path("ml/ocr/recognizer_confirmed_acceptance_v9/protocol.py"),
    Path("ml/ocr/recognizer_confirmed_acceptance_v9/dataset.py"),
    Path("ml/ocr/recognizer_confirmed_acceptance_v9/prepare_split.py"),
    Path("ml/ocr/recognizer_confirmed_acceptance_v9/PROTOCOL.json"),
    Path("src/GraphReader.Ocr/OcrV8ProductionCompositionPipeline.cs"),
    Path("src/GraphReader.Ocr/OcrV9CandidateCompositionFactory.cs"),
    Path("tests/GraphReader.Ocr.Tests/OcrV9CandidateSelectionTests.cs"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    seal_output = args.seal_output.resolve()
    if output.exists() or seal_output.exists():
        raise RuntimeError("V9 selection archive or seal already exists")
    if configuration()["state"] != "p1_preregistered_before_selection_identity_materialization_or_model_execution":
        raise RuntimeError("V9 P1 protocol state changed")
    secret_seed = secrets.randbits(128)
    summary = save_selection_archive(build_selection_split(secret_seed), output)
    seal = {
        **summary,
        "schema": "graphreader.ocr-recognizer-confirmed-selection-seal.v1",
        "candidate_id": "P1",
        "fixture_archive_path": output.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": {
            path.as_posix(): sha256_file(REPO_ROOT / path)
            for path in SOURCE_PATHS
        },
        "generation_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "secret_seed_serialized": False,
        "model_execution_count_at_freeze": 0,
        "selection_execution_authorized": False,
        "public_gate_authorized": False,
        "public_gate_evaluations": 0,
        "production_approval": False,
        "release_eligible": False,
    }
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    seal_output.write_bytes(canonical_json_bytes(seal))
    print(f"fixture_archive_sha256={summary['fixture_archive_sha256']}")
    print(f"fixture_manifest_sha256={summary['fixture_manifest_sha256']}")
    print(f"split_seal_sha256={sha256_file(seal_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
