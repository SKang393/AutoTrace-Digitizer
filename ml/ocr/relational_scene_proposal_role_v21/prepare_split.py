# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze OCR V21 train, selection, and sealed-public identities once."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_file
from .dataset import build_split, save_archive
from .protocol import protocol_configuration


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = Path("ml/ocr/relational_scene_proposal_role_v21/PROTOCOL.json")
TRIGGER_RESULT_PATH = Path("ml/ocr/cross_model_consensus_v9_p3/P3_SELECTION_RESULT.json")
SOURCE_PATHS = (
    PROTOCOL_PATH,
    Path("ml/ocr/relational_scene_proposal_role_v21/protocol.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/dataset.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/model.py"),
    Path("ml/ocr/relational_scene_proposal_role_v21/prepare_split.py"),
    TRIGGER_RESULT_PATH,
    Path("ml/ocr/layout_conditioned_proposal_role_v15/dataset.py"),
    Path("ml/ocr/layout_conditioned_proposal_role_v15/protocol.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/protocol.py"),
    Path("ml/markers/gate_seal.py"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-Medium.ttf"),
    Path("src/GraphReader.App/Assets/Fonts/NotoSans-SemiBold.ttf"),
)


def freeze_identities(train_output: Path, selection_output: Path, public_output: Path, seal_output: Path) -> dict[str, object]:
    for path in (train_output, selection_output, public_output, seal_output):
        if path.exists():
            raise FileExistsError(f"OCR V21 identity exists; refusing regeneration: {path}")
    configuration = protocol_configuration()
    if (REPO_ROOT / PROTOCOL_PATH).read_bytes() != canonical_json_bytes(configuration):
        raise RuntimeError("OCR V21 committed protocol is not canonical")
    expected_trigger_sha256 = configuration["predecessor_aggregate_only"]["p3_selection_result_sha256"]
    if sha256_file(REPO_ROOT / TRIGGER_RESULT_PATH) != expected_trigger_sha256:
        raise RuntimeError("OCR V21 aggregate-only trigger result changed before identity freeze")
    train = save_archive(build_split("train"), "train", train_output)
    selection = save_archive(build_split("validation"), "validation", selection_output)
    public = save_archive(build_split("sealed_public"), "sealed_public", public_output)
    source_sha256 = {path.as_posix(): sha256_file(REPO_ROOT / path) for path in SOURCE_PATHS}
    seal: dict[str, object] = {
        "schema": "graphreader.ocr-relational-scene-proposal-role-split-seal.v1",
        "candidate_id": "P1",
        "train": train,
        "selection": selection,
        "sealed_public": public,
        "source_sha256": source_sha256,
        "optimizer_steps_at_freeze": 0,
        "selection_evaluations": 0,
        "public_evaluations": 0,
        "training_authorized": False,
        "public_execution_authorized": False,
        "marker_creation_evaluated": False,
        "artifact_mask_production_approval": False,
        "production_approval": False,
        "release_eligible": False,
    }
    seal_output.parent.mkdir(parents=True, exist_ok=True)
    seal_output.write_bytes(canonical_json_bytes(seal))
    return seal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--selection-output", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--seal-output", type=Path, required=True)
    arguments = parser.parse_args()
    seal = freeze_identities(
        arguments.train_output.resolve(),
        arguments.selection_output.resolve(),
        arguments.public_output.resolve(),
        arguments.seal_output.resolve(),
    )
    print(json.dumps({
        "train_archive_sha256": seal["train"]["archive_sha256"],
        "selection_archive_sha256": seal["selection"]["archive_sha256"],
        "public_archive_sha256": seal["sealed_public"]["archive_sha256"],
        "seal_sha256": sha256(canonical_json_bytes(seal)).hexdigest(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
