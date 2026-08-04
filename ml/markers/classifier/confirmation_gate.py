# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Independent confirmation split frozen after validation-only model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from .dataset import (
    ARTIFACT_KINDS,
    FILL_NAMES,
    PATCH_SIZE,
    SCENARIOS,
    SHAPE_NAMES,
    PatchSample,
    _draw_artifact,
    _draw_context,
    _draw_marker,
)
from .public_gate import (
    MARKER_CLASSIFIER_TASK,
    ClassifierGateDefinition,
    _evaluate_gate,
)


CONFIRMATION_REVISION = "marker-classifier-confirmation-gate-v2"
CONFIRMATION_SEED = 20260824
CONFIRMATION_CONFIG_PATH = Path("ml/markers/classifier/gates/confirmation-v2.json")
CLASSIFIER_CONFIRMATION_GATE = ClassifierGateDefinition(
    task=MARKER_CLASSIFIER_TASK,
    revision=CONFIRMATION_REVISION,
    split_config_path=CONFIRMATION_CONFIG_PATH,
    evaluator_source_paths=(
        Path("ml/markers/classifier/public_gate.py"),
        Path("ml/markers/classifier/confirmation_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/classifier/dataset.py"),
        Path("ml/markers/classifier/metrics.py"),
        Path("ml/markers/classifier/export.py"),
        Path("ml/markers/classifier/model.py"),
    ),
)


def _geometry(template: str, repeat: int) -> tuple[float, float, float, float]:
    values = {
        "upper_left_oblong": (14.9, 14.6, 7.2, 0.84),
        "lower_right_compact": (17.1, 17.3, 6.5, 1.08),
    }
    cx, cy, radius, aspect = values[template]
    return cx + 0.15 * repeat, cy - 0.12 * repeat, radius, aspect


def _degrade(image: Image.Image, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "microfilm_dither":
        image = image.resize((27, 27), Image.Resampling.BILINEAR).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
    elif family == "laser_streak":
        image = image.filter(ImageFilter.GaussianBlur(0.32))
    else:
        raise ValueError(family)
    luminance = np.asarray(image, dtype=np.float32) / 255.0
    if family == "microfilm_dither":
        selector = rng.random(luminance.shape)
        luminance = np.where(selector < 0.0015, 0.0, np.where(selector > 0.9985, 1.0, luminance))
    else:
        luminance[:, 5::13] = np.clip(luminance[:, 5::13] - 0.018, 0.0, 1.0)
    luminance = np.clip(luminance + rng.normal(0.0, 0.005, luminance.shape), 0.0, 1.0)
    return (1.0 - luminance).astype(np.float32)


def _marker_sample(
    family: str,
    template: str,
    shape_index: int,
    fill_index: int,
    repeat: int,
    seed: int,
) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    geometry = _geometry(template, repeat)
    scenario = SCENARIOS[(shape_index * len(FILL_NAMES) + fill_index + repeat + 3) % len(SCENARIOS)]
    if shape_index in (SHAPE_NAMES.index("star"), SHAPE_NAMES.index("asterisk"), SHAPE_NAMES.index("cross")):
        scenario = "minority_probe"
    _draw_context(draw, scenario, geometry[0], geometry[1], scale)
    _draw_marker(draw, SHAPE_NAMES[shape_index], FILL_NAMES[fill_index], geometry, scale, repeat + 8)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"confirm-v2-{family}-{template}-{SHAPE_NAMES[shape_index]}-{FILL_NAMES[fill_index]}-{repeat}"
    return PatchSample(sample_id, "confirmation_gate", family, template, scenario, tensor, shape_index, fill_index, 0.0, None)


def _artifact_sample(family: str, template: str, kind: str, repeat: int, seed: int) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    _draw_artifact(draw, kind, scale, repeat + 5)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"confirm-v2-{family}-{template}-artifact-{kind}-{repeat}"
    return PatchSample(
        sample_id,
        "confirmation_gate",
        family,
        template,
        kind,
        tensor,
        SHAPE_NAMES.index("other"),
        FILL_NAMES.index("unknown"),
        1.0,
        kind,
    )


def build_confirmation_split() -> tuple[PatchSample, ...]:
    samples: list[PatchSample] = []
    ordinal = 0
    for family, template in (("microfilm_dither", "upper_left_oblong"), ("laser_streak", "lower_right_compact")):
        for repeat in range(2):
            for shape_index in range(len(SHAPE_NAMES)):
                for fill_index in range(len(FILL_NAMES)):
                    samples.append(
                        _marker_sample(
                            family,
                            template,
                            shape_index,
                            fill_index,
                            repeat,
                            CONFIRMATION_SEED + ordinal,
                        )
                    )
                    ordinal += 1
            for kind in ARTIFACT_KINDS:
                samples.append(_artifact_sample(family, template, kind, repeat, CONFIRMATION_SEED + ordinal))
                ordinal += 1
    return tuple(samples)


def confirmation_manifest(samples: tuple[PatchSample, ...]) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "task": MARKER_CLASSIFIER_TASK,
        "revision": CONFIRMATION_REVISION,
        "seed": CONFIRMATION_SEED,
        "selection_use": False,
        "private_data": False,
        "frozen_after_model_selection": True,
        "cases": [
            {
                "sample_id": sample.sample_id,
                "family": sample.family,
                "template": sample.template,
                "scenario": sample.scenario,
                "shape": SHAPE_NAMES[sample.shape_index],
                "fill": FILL_NAMES[sample.fill_index],
                "artifact": sample.artifact,
                "artifact_kind": sample.artifact_kind,
                "tensor_sha256": hashlib.sha256(sample.tensor.numpy().tobytes(order="C")).hexdigest(),
            }
            for sample in samples
        ],
    }


def evaluate_confirmation_gate(checkpoint: Path, onnx_path: Path, output_dir: Path) -> dict[str, object]:
    samples = build_confirmation_split()
    return _evaluate_gate(
        checkpoint,
        onnx_path,
        output_dir,
        definition=CLASSIFIER_CONFIRMATION_GATE,
        samples=samples,
        manifest_payload=confirmation_manifest(samples),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_confirmation_gate(args.checkpoint, args.onnx, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
