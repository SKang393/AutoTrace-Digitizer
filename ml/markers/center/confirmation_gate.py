# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Independent confirmation split frozen after validation-only model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .public_gate import (
    MARKER_CENTER_TASK,
    CenterGateDefinition,
    PublicGateScene,
    _evaluate_gate,
    _render_scene,
)


CONFIRMATION_REVISION = "marker-center-confirmation-gate-v1"
CONFIRMATION_SEED = 20260814
CONFIRMATION_CONFIG_PATH = Path("ml/markers/center/gates/confirmation-v1.json")
CENTER_CONFIRMATION_GATE = CenterGateDefinition(
    task=MARKER_CENTER_TASK,
    revision=CONFIRMATION_REVISION,
    split_config_path=CONFIRMATION_CONFIG_PATH,
    evaluator_source_paths=(
        Path("ml/markers/center/public_gate.py"),
        Path("ml/markers/center/confirmation_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/center/dataset.py"),
        Path("ml/markers/center/metrics.py"),
        Path("ml/markers/center/postprocess.py"),
    ),
)


def build_confirmation_split() -> tuple[PublicGateScene, ...]:
    definitions = (
        ("confirm-rise", "fax_resample", "low_rising_arc", ((18, 73), (36, 61), (55, 49), (74, 38), (93, 29), (110, 44))),
        ("confirm-switch", "toner_gamma", "high_low_switch", ((17, 34), (35, 63), (53, 37), (71, 66), (89, 41), (107, 70))),
        ("confirm-clusters", "vertical_banding", "separated_clusters", ((20, 55), (36, 39), (52, 60), (76, 31), (92, 52), (108, 35))),
    )
    return tuple(
        _render_scene(scene_id, family, template, centers, CONFIRMATION_SEED + index)
        for index, (scene_id, family, template, centers) in enumerate(definitions)
    )


def confirmation_manifest(scenes: tuple[PublicGateScene, ...]) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "task": MARKER_CENTER_TASK,
        "revision": CONFIRMATION_REVISION,
        "seed": CONFIRMATION_SEED,
        "selection_use": False,
        "private_data": False,
        "frozen_after_model_selection": True,
        "cases": [
            {
                "scene_id": scene.scene_id,
                "family": scene.family,
                "template": scene.template,
                "tensor_sha256": hashlib.sha256(scene.tensor.tobytes(order="C")).hexdigest(),
                "center_count": len(scene.centers),
                "centers": scene.centers,
                "hard_negative_kinds": [item[0] for item in scene.hard_negatives],
            }
            for scene in scenes
        ],
    }


def evaluate_confirmation_gate(onnx_path: Path, output_dir: Path) -> dict[str, object]:
    scenes = build_confirmation_split()
    return _evaluate_gate(
        onnx_path,
        output_dir,
        definition=CENTER_CONFIRMATION_GATE,
        scenes=scenes,
        manifest_payload=confirmation_manifest(scenes),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_confirmation_gate(args.onnx, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
