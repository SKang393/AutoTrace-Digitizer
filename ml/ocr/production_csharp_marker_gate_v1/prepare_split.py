# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Generate and seal the new public archive without executing any model."""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import secrets

import numpy as np
import PIL
import torch

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import build_sealed_split, save_sealed_archive
from .protocol import protocol_configuration


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PATHS = (
    Path("ml/markers/center/runtime_consistency_v2/dataset.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/ocr/production_csharp_marker_gate_v1/protocol.py"),
    Path("ml/ocr/production_csharp_marker_gate_v1/dataset.py"),
    Path("ml/ocr/production_csharp_marker_gate_v1/prepare_split.py"),
    Path("ml/ocr/production_csharp_marker_gate_v1/PROTOCOL.json"),
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
    Path("tests/GraphReader.Integration.Tests/IntegrationSmoke/OcrMarkerDirectPublicGateTests.cs"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    seal_output = args.seal_output.resolve()
    if output.exists() or seal_output.exists():
        raise RuntimeError("A frozen fixture archive or split seal already exists")
    if protocol_configuration()["state"] != "frozen_before_any_model_execution_on_the_new_split":
        raise RuntimeError("The gate protocol is not in its preregistered state")

    secret_seed = secrets.randbits(128)
    scenes = build_sealed_split(secret_seed)
    summary = save_sealed_archive(scenes, output)
    source_hashes = {
        path.as_posix(): sha256_file(REPO_ROOT / path)
        for path in SOURCE_PATHS
    }
    seal = {
        **summary,
        "schema": "graphreader.ocr-marker-production-composition-split-seal.v1",
        "fixture_archive_path": output.relative_to(REPO_ROOT).as_posix(),
        "source_sha256": source_hashes,
        "generation_environment": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "model_execution_count_at_freeze": 0,
        "secret_seed_serialized": False,
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
