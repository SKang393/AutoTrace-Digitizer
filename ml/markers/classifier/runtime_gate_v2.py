# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen single-use public and confirmation gates for runtime contract v2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_evaluator_identity,
)
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
from .metrics import binary_metrics, classification_metrics
from .runtime_v2 import PARITY_TOLERANCE, RUNTIME_V2_OUTPUT_NAME, run_probability_runtime


MARKER_CLASSIFIER_TASK = "marker-classifier"
PUBLIC_REVISION = "marker-classifier-public-gate-v3"
CONFIRMATION_REVISION = "marker-classifier-confirmation-gate-v3"
PUBLIC_SEED = 20260831
CONFIRMATION_SEED = 20260901
PUBLIC_CONFIG_PATH = Path("ml/markers/classifier/gates/public-v3.json")
CONFIRMATION_CONFIG_PATH = Path("ml/markers/classifier/gates/confirmation-v3.json")
SHAPE_MACRO_F1_GATE = 0.90
FILL_MACRO_F1_GATE = 0.90
ARTIFACT_F1_GATE = 1.0
MINORITY_CLASS_F1_GATE = 0.90
RUNTIME_GATE_CONFIG = {
    "shape_macro_f1": SHAPE_MACRO_F1_GATE,
    "fill_macro_f1": FILL_MACRO_F1_GATE,
    "artifact_f1": ARTIFACT_F1_GATE,
    "minority_class_f1": MINORITY_CLASS_F1_GATE,
    "probability_packed_onnx_parity": PARITY_TOLERANCE,
}


@dataclass(frozen=True)
class RuntimeGateDefinition:
    task: str
    revision: str
    seed: int
    split_config_path: Path
    manifest_path: Path
    split_name: str
    families_and_templates: tuple[tuple[str, str], ...]

    @property
    def evaluator_source_paths(self) -> tuple[Path, ...]:
        return (
            Path("ml/markers/classifier/runtime_gate_v2.py"),
            Path("ml/markers/classifier/runtime_v2.py"),
            Path("ml/markers/gate_seal.py"),
            Path("ml/markers/classifier/dataset.py"),
            Path("ml/markers/classifier/metrics.py"),
            Path("ml/markers/classifier/model.py"),
            self.manifest_path,
        )


PUBLIC_GATE = RuntimeGateDefinition(
    MARKER_CLASSIFIER_TASK,
    PUBLIC_REVISION,
    PUBLIC_SEED,
    PUBLIC_CONFIG_PATH,
    Path("ml/markers/classifier/manifests/public-v3.json"),
    "public_gate_v3",
    (("newspaper_bleed", "upper_right_square"), ("moire_reduction", "lower_left_wide")),
)
CONFIRMATION_GATE = RuntimeGateDefinition(
    MARKER_CLASSIFIER_TASK,
    CONFIRMATION_REVISION,
    CONFIRMATION_SEED,
    CONFIRMATION_CONFIG_PATH,
    Path("ml/markers/classifier/manifests/confirmation-v3.json"),
    "confirmation_gate_v3",
    (("dot_matrix_echo", "left_center_tall"), ("scanner_shadow", "right_center_compact")),
)


def _geometry(template: str, repeat: int) -> tuple[float, float, float, float]:
    values = {
        "upper_right_square": (16.9, 15.0, 6.7, 0.98),
        "lower_left_wide": (15.0, 17.0, 7.2, 0.86),
        "left_center_tall": (15.1, 16.1, 6.9, 1.10),
        "right_center_compact": (17.0, 15.9, 6.4, 0.96),
    }
    cx, cy, radius, aspect = values[template]
    return cx + repeat * 0.14, cy - repeat * 0.11, radius, aspect


def _degrade(image: Image.Image, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "newspaper_bleed":
        image = image.filter(ImageFilter.GaussianBlur(0.38)).filter(ImageFilter.MinFilter(3))
    elif family == "moire_reduction":
        image = image.resize((28, 28), Image.Resampling.BILINEAR).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
    elif family == "dot_matrix_echo":
        image = image.resize((30, 30), Image.Resampling.NEAREST).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BILINEAR)
    elif family == "scanner_shadow":
        image = image.filter(ImageFilter.GaussianBlur(0.27))
    else:
        raise ValueError(family)
    luminance = np.asarray(image, dtype=np.float32) / 255.0
    if family == "newspaper_bleed":
        luminance = np.power(luminance, 0.97)
    elif family == "moire_reduction":
        luminance[::7, :] = np.clip(luminance[::7, :] + 0.012, 0.0, 1.0)
    elif family == "dot_matrix_echo":
        selector = np.indices(luminance.shape).sum(axis=0) % 9 == 0
        luminance = np.where(selector, np.clip(luminance + 0.009, 0.0, 1.0), luminance)
    elif family == "scanner_shadow":
        luminance = np.clip(luminance + np.linspace(0.012, -0.006, PATCH_SIZE, dtype=np.float32)[:, None], 0.0, 1.0)
    luminance = np.clip(luminance + rng.normal(0.0, 0.0048, luminance.shape), 0.0, 1.0)
    return (1.0 - luminance).astype(np.float32)


def _marker_sample(
    definition: RuntimeGateDefinition,
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
    scenario = SCENARIOS[(shape_index * len(FILL_NAMES) + fill_index + repeat + 1) % len(SCENARIOS)]
    if shape_index in (SHAPE_NAMES.index("star"), SHAPE_NAMES.index("asterisk"), SHAPE_NAMES.index("cross")):
        scenario = "minority_probe"
    _draw_context(draw, scenario, geometry[0], geometry[1], scale)
    _draw_marker(draw, SHAPE_NAMES[shape_index], FILL_NAMES[fill_index], geometry, scale, repeat + 12)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"{definition.split_name}-{family}-{template}-{SHAPE_NAMES[shape_index]}-{FILL_NAMES[fill_index]}-{repeat}"
    return PatchSample(
        sample_id,
        definition.split_name,
        family,
        template,
        scenario,
        tensor,
        shape_index,
        fill_index,
        0.0,
        None,
    )


def _artifact_sample(
    definition: RuntimeGateDefinition,
    family: str,
    template: str,
    kind: str,
    repeat: int,
    seed: int,
) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    _draw_artifact(draw, kind, scale, repeat + 9)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"{definition.split_name}-{family}-{template}-artifact-{kind}-{repeat}"
    return PatchSample(
        sample_id,
        definition.split_name,
        family,
        template,
        kind,
        tensor,
        SHAPE_NAMES.index("other"),
        FILL_NAMES.index("unknown"),
        1.0,
        kind,
    )


def build_gate_split(definition: RuntimeGateDefinition) -> tuple[PatchSample, ...]:
    samples: list[PatchSample] = []
    ordinal = 0
    for family, template in definition.families_and_templates:
        for repeat in range(2):
            for shape_index in range(len(SHAPE_NAMES)):
                for fill_index in range(len(FILL_NAMES)):
                    samples.append(
                        _marker_sample(
                            definition,
                            family,
                            template,
                            shape_index,
                            fill_index,
                            repeat,
                            definition.seed + ordinal,
                        )
                    )
                    ordinal += 1
            for kind in ARTIFACT_KINDS:
                samples.append(
                    _artifact_sample(definition, family, template, kind, repeat, definition.seed + ordinal)
                )
                ordinal += 1
    return tuple(samples)


def gate_manifest(definition: RuntimeGateDefinition) -> dict[str, object]:
    return {
        "manifest_version": 1,
        "task": definition.task,
        "revision": definition.revision,
        "seed": definition.seed,
        "selection_use": False,
        "private_data": False,
        "frozen_before_candidate_execution": True,
        "families": [family for family, _ in definition.families_and_templates],
        "templates": [template for _, template in definition.families_and_templates],
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
            for sample in build_gate_split(definition)
        ],
    }


def classifier_gate_results(
    *,
    shape_macro_f1: float,
    fill_macro_f1: float,
    artifact_f1: float,
    minority_shape_f1: dict[str, float],
    parity_maximum_absolute_error: float,
) -> dict[str, bool]:
    return {
        "shape_macro_f1": shape_macro_f1 >= SHAPE_MACRO_F1_GATE,
        "fill_macro_f1": fill_macro_f1 >= FILL_MACRO_F1_GATE,
        "artifact_f1": math.isclose(artifact_f1, ARTIFACT_F1_GATE, abs_tol=1e-12),
        "minority_shape_preservation": min(minority_shape_f1.values()) >= MINORITY_CLASS_F1_GATE,
        "probability_packed_onnx_parity": parity_maximum_absolute_error <= PARITY_TOLERANCE,
    }


def evaluate_gate(
    checkpoint: Path,
    onnx_path: Path,
    output_dir: Path,
    definition: RuntimeGateDefinition,
) -> dict[str, object]:
    local_seal_path = output_dir / "evaluation.seal.json"
    if local_seal_path.exists():
        raise RuntimeError(f"Gate was already opened; refusing a repeated evaluation: {local_seal_path}")
    manifest = gate_manifest(definition)
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / definition.manifest_path).read_bytes() != manifest_bytes:
        raise RuntimeError("Generated gate manifest does not match the committed frozen manifest")
    split_config = json.loads((repo_root / definition.split_config_path).read_text(encoding="utf-8"))
    require_evaluator_identity(
        expected_task=definition.task,
        expected_revision=definition.revision,
        manifest=manifest,
        split_config=split_config,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    onnx_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    seal = acquire_gate_seal(
        repo_root=repo_root,
        task=definition.task,
        revision=definition.revision,
        candidate_hashes={"checkpoint_sha256": checkpoint_sha256, "probability_packed_onnx_sha256": onnx_sha256},
        dataset_manifest_sha256=manifest_sha256,
        split_config_path=definition.split_config_path,
        evaluator_source_paths=definition.evaluator_source_paths,
        gate_config=RUNTIME_GATE_CONFIG,
    )
    require_evaluator_identity(
        expected_task=definition.task,
        expected_revision=definition.revision,
        manifest=manifest,
        split_config=split_config,
        seal_binding=seal.binding,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate-manifest.json").write_bytes(manifest_bytes)
    local_seal_path.write_bytes(
        canonical_json_bytes(
            {
                "status": "opened",
                "evaluation_count": 1,
                "canonical_seal_key": seal.key,
                "canonical_opened_path": str(seal.opened_path),
                "binding": seal.binding,
            }
        )
    )
    started = time.perf_counter()
    try:
        samples = build_gate_split(definition)
        _, actual, parity_max, inference_ms, provider = run_probability_runtime(checkpoint, onnx_path, samples)
        marker_indices = np.array([index for index, sample in enumerate(samples) if sample.artifact < 0.5])
        shape_targets = np.array([sample.shape_index for sample in samples], dtype=np.int64)[marker_indices]
        fill_targets = np.array([sample.fill_index for sample in samples], dtype=np.int64)[marker_indices]
        artifact_targets = np.array([sample.artifact for sample in samples], dtype=np.float32)
        shape = classification_metrics(actual[marker_indices, 0:9], shape_targets, len(SHAPE_NAMES))
        fill = classification_metrics(actual[marker_indices, 9:12], fill_targets, len(FILL_NAMES))
        artifact = binary_metrics(actual[:, 12], artifact_targets)
        minority = {
            name: shape.per_class_f1[SHAPE_NAMES.index(name)]
            for name in ("star", "asterisk", "cross")
        }
        gate_results = classifier_gate_results(
            shape_macro_f1=shape.macro_f1,
            fill_macro_f1=fill.macro_f1,
            artifact_f1=float(artifact["f1"]),
            minority_shape_f1=minority,
            parity_maximum_absolute_error=parity_max,
        )
        report: dict[str, object] = {
            "status": "pass" if all(gate_results.values()) else "fail",
            "release_eligible": False,
            "release_blocker": "Model resolution, production composition, packaging, and clean-machine evidence remain separate mandatory gates.",
            "task": definition.task,
            "revision": definition.revision,
            "evaluation_count": 1,
            "dataset_manifest_sha256": manifest_sha256,
            "sample_count": len(samples),
            "marker_sample_count": len(marker_indices),
            "checkpoint_sha256": checkpoint_sha256,
            "probability_packed_onnx_sha256": onnx_sha256,
            "canonical_seal_key": seal.key,
            "seal_binding": seal.binding,
            "provider": provider,
            "runtime_output_name": RUNTIME_V2_OUTPUT_NAME,
            "runtime_output_shape": list(actual.shape),
            "metrics": {
                "shape": shape.to_dict(),
                "fill": fill.to_dict(),
                "artifact": artifact,
                "minority_shape_f1": minority,
            },
            "gates": RUNTIME_GATE_CONFIG,
            "gate_results": gate_results,
            "probability_packed_onnx_maximum_absolute_error": parity_max,
            "inference_total_ms": round(inference_ms, 3),
            "inference_ms_per_patch": round(inference_ms / len(samples), 6),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        require_evaluator_identity(
            expected_task=definition.task,
            expected_revision=definition.revision,
            manifest=manifest,
            split_config=split_config,
            seal_binding=seal.binding,
            report=report,
        )
        report_path = output_dir / "gate-report.json"
        report_path.write_bytes(canonical_json_bytes(report))
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        result_path = complete_gate_seal(seal, status=str(report["status"]), report_sha256=report_sha256)
        local_seal_path.write_bytes(
            canonical_json_bytes(
                {
                    "status": "completed",
                    "evaluation_count": 1,
                    "manifest_sha256": manifest_sha256,
                    "report_sha256": report_sha256,
                    "canonical_seal_key": seal.key,
                    "canonical_result_path": str(result_path),
                    "binding": seal.binding,
                }
            )
        )
        return report
    except Exception as error:
        local_seal_path.write_bytes(
            canonical_json_bytes(
                {"status": "failed_after_open", "evaluation_count": 1, "error_type": type(error).__name__}
            )
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", choices=("public", "confirmation"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    definition = PUBLIC_GATE if args.gate == "public" else CONFIRMATION_GATE
    report = evaluate_gate(args.checkpoint, args.onnx, args.output, definition)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_F1_GATE",
    "CONFIRMATION_GATE",
    "FILL_MACRO_F1_GATE",
    "MINORITY_CLASS_F1_GATE",
    "PUBLIC_GATE",
    "RUNTIME_GATE_CONFIG",
    "SHAPE_MACRO_F1_GATE",
    "build_gate_split",
    "classifier_gate_results",
    "evaluate_gate",
    "gate_manifest",
]
