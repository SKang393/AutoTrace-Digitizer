# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use direct packed-runtime gate for the marker classifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
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
from .export import PackedRuntimeClassifier, RUNTIME_OUTPUT_NAME
from .metrics import binary_metrics, classification_metrics
from .model import load_checkpoint


PUBLIC_GATE_REVISION = "marker-classifier-public-gate-v1"
PUBLIC_GATE_SEED = 20260804
SHAPE_MACRO_F1_GATE = 0.90
FILL_MACRO_F1_GATE = 0.90
ARTIFACT_F1_GATE = 1.0
MINORITY_CLASS_F1_GATE = 0.90
PARITY_TOLERANCE = 1e-5
PUBLIC_GATE_CONFIG_PATH = Path("ml/markers/classifier/gates/public-v1.json")
PUBLIC_GATE_CONFIG = {
    "shape_macro_f1": SHAPE_MACRO_F1_GATE,
    "fill_macro_f1": FILL_MACRO_F1_GATE,
    "artifact_f1": ARTIFACT_F1_GATE,
    "minority_class_f1": MINORITY_CLASS_F1_GATE,
    "packed_onnx_parity": PARITY_TOLERANCE,
}
MARKER_CLASSIFIER_TASK = "marker-classifier"


@dataclass(frozen=True)
class ClassifierGateDefinition:
    task: str
    revision: str
    split_config_path: Path
    evaluator_source_paths: tuple[Path, ...]


CLASSIFIER_PUBLIC_GATE = ClassifierGateDefinition(
    task=MARKER_CLASSIFIER_TASK,
    revision=PUBLIC_GATE_REVISION,
    split_config_path=PUBLIC_GATE_CONFIG_PATH,
    evaluator_source_paths=(
        Path("ml/markers/classifier/public_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/classifier/dataset.py"),
        Path("ml/markers/classifier/metrics.py"),
        Path("ml/markers/classifier/export.py"),
        Path("ml/markers/classifier/model.py"),
    ),
)


def _geometry(template: str, repeat: int) -> tuple[float, float, float, float]:
    values = {
        "right_high": (16.35, 15.35, 6.85, 0.93),
        "left_low": (15.45, 16.65, 7.15, 1.02),
    }
    cx, cy, radius, aspect = values[template]
    return cx + 0.18 * repeat, cy - 0.16 * repeat, radius, aspect


def _degrade(image: Image.Image, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "fax_resample":
        image = image.resize((29, 29), Image.Resampling.BILINEAR).resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.BICUBIC)
    elif family == "toner_gamma":
        image = image.filter(ImageFilter.GaussianBlur(0.20))
    else:
        raise ValueError(family)
    luminance = np.asarray(image, dtype=np.float32) / 255.0
    if family == "toner_gamma":
        luminance = np.power(luminance, 1.06)
    luminance = np.clip(luminance + rng.normal(0.0, 0.0045, luminance.shape), 0.0, 1.0)
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
    scenario = SCENARIOS[(shape_index * len(FILL_NAMES) + fill_index + repeat + 2) % len(SCENARIOS)]
    if shape_index in (SHAPE_NAMES.index("star"), SHAPE_NAMES.index("asterisk"), SHAPE_NAMES.index("cross")):
        scenario = "minority_probe"
    _draw_context(draw, scenario, geometry[0], geometry[1], scale)
    _draw_marker(draw, SHAPE_NAMES[shape_index], FILL_NAMES[fill_index], geometry, scale, repeat + 4)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"public-{family}-{template}-{SHAPE_NAMES[shape_index]}-{FILL_NAMES[fill_index]}-{repeat}"
    return PatchSample(sample_id, "public_gate", family, template, scenario, tensor, shape_index, fill_index, 0.0, None)


def _artifact_sample(family: str, template: str, kind: str, repeat: int, seed: int) -> PatchSample:
    scale = 3
    image = Image.new("L", (PATCH_SIZE * scale, PATCH_SIZE * scale), 255)
    draw = ImageDraw.Draw(image)
    _draw_artifact(draw, kind, scale, repeat + 2)
    image = image.resize((PATCH_SIZE, PATCH_SIZE), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(_degrade(image, family, seed)[None].copy())
    sample_id = f"public-{family}-{template}-artifact-{kind}-{repeat}"
    return PatchSample(
        sample_id,
        "public_gate",
        family,
        template,
        kind,
        tensor,
        SHAPE_NAMES.index("other"),
        FILL_NAMES.index("unknown"),
        1.0,
        kind,
    )


def build_public_gate_split() -> tuple[PatchSample, ...]:
    samples: list[PatchSample] = []
    ordinal = 0
    for family, template in (("fax_resample", "right_high"), ("toner_gamma", "left_low")):
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
                            PUBLIC_GATE_SEED + ordinal,
                        )
                    )
                    ordinal += 1
            for kind in ARTIFACT_KINDS:
                samples.append(_artifact_sample(family, template, kind, repeat, PUBLIC_GATE_SEED + ordinal))
                ordinal += 1
    return tuple(samples)


def public_gate_manifest() -> dict[str, object]:
    cases = []
    for sample in build_public_gate_split():
        cases.append(
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
        )
    return {
        "manifest_version": 1,
        "task": MARKER_CLASSIFIER_TASK,
        "revision": PUBLIC_GATE_REVISION,
        "seed": PUBLIC_GATE_SEED,
        "selection_use": False,
        "private_data": False,
        "cases": cases,
    }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


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
        "packed_onnx_parity": parity_maximum_absolute_error <= PARITY_TOLERANCE,
    }


def _evaluate_gate(
    checkpoint: Path,
    onnx_path: Path,
    output_dir: Path,
    *,
    definition: ClassifierGateDefinition,
    samples: tuple[PatchSample, ...],
    manifest_payload: dict[str, object],
) -> dict[str, object]:
    selected_samples = samples
    seal_path = output_dir / "evaluation.seal.json"
    if seal_path.exists():
        raise RuntimeError(f"Public gate was already opened; refusing a repeated evaluation: {seal_path}")
    manifest_bytes = canonical_json_bytes(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    repo_root = Path(__file__).resolve().parents[3]
    split_config = json.loads((repo_root / definition.split_config_path).read_text(encoding="utf-8"))
    require_evaluator_identity(
        expected_task=definition.task,
        expected_revision=definition.revision,
        manifest=manifest_payload,
        split_config=split_config,
    )
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    candidate_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    canonical_seal = acquire_gate_seal(
        repo_root=repo_root,
        task=definition.task,
        revision=definition.revision,
        candidate_hashes={
            "checkpoint_sha256": checkpoint_sha256,
            "packed_onnx_sha256": candidate_sha256,
        },
        dataset_manifest_sha256=manifest_sha256,
        split_config_path=definition.split_config_path,
        evaluator_source_paths=definition.evaluator_source_paths,
        gate_config=PUBLIC_GATE_CONFIG,
    )
    require_evaluator_identity(
        expected_task=definition.task,
        expected_revision=definition.revision,
        manifest=manifest_payload,
        split_config=split_config,
        seal_binding=canonical_seal.binding,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "public-gate-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    _write_json(
        seal_path,
        {
            "status": "opened",
            "evaluation_count": 1,
            "canonical_seal_key": canonical_seal.key,
            "canonical_opened_path": str(canonical_seal.opened_path),
            "binding": canonical_seal.binding,
        },
    )
    started = time.perf_counter()
    try:
        samples = selected_samples
        session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
        model, payload = load_checkpoint(checkpoint)
        packed_model = PackedRuntimeClassifier(
            model,
            float(payload["shape_temperature"]),
            float(payload["fill_temperature"]),
        ).eval()
        packed_rows = []
        parity_max = 0.0
        inference_ms = 0.0
        for start in range(0, len(samples), 64):
            tensor = torch.stack([sample.tensor for sample in samples[start : start + 64]])
            inference_started = time.perf_counter()
            actual = session.run([RUNTIME_OUTPUT_NAME], {model.contract.input_name: tensor.numpy()})[0]
            inference_ms += (time.perf_counter() - inference_started) * 1000.0
            with torch.inference_mode():
                expected = packed_model(tensor).numpy()
            parity_max = max(parity_max, float(np.max(np.abs(actual - expected))))
            packed_rows.append(actual)
        packed = np.concatenate(packed_rows, axis=0)
        marker_indices = np.array([index for index, sample in enumerate(samples) if sample.artifact < 0.5])
        shape_targets = np.array([sample.shape_index for sample in samples], dtype=np.int64)[marker_indices]
        fill_targets = np.array([sample.fill_index for sample in samples], dtype=np.int64)[marker_indices]
        artifact_targets = np.array([sample.artifact for sample in samples], dtype=np.float32)
        shape = classification_metrics(_softmax(packed[marker_indices, 0:9]), shape_targets, len(SHAPE_NAMES))
        fill = classification_metrics(_softmax(packed[marker_indices, 9:12]), fill_targets, len(FILL_NAMES))
        artifact = binary_metrics(_sigmoid(packed[:, 12]), artifact_targets)
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
        report = {
            "status": "pass" if all(gate_results.values()) else "fail",
            "release_eligible": False,
            "release_blocker": "Production runtime discovery and packaging evidence are outside this marker-owned gate.",
            "task": definition.task,
            "revision": definition.revision,
            "evaluation_count": 1,
            "dataset_manifest_sha256": manifest_sha256,
            "sample_count": len(samples),
            "marker_sample_count": len(marker_indices),
            "checkpoint_sha256": checkpoint_sha256,
            "packed_onnx_sha256": candidate_sha256,
            "canonical_seal_key": canonical_seal.key,
            "seal_binding": canonical_seal.binding,
            "provider": session.get_providers()[0],
            "runtime_output_name": RUNTIME_OUTPUT_NAME,
            "runtime_output_shape": list(packed.shape),
            "metrics": {
                "shape": shape.to_dict(),
                "fill": fill.to_dict(),
                "artifact": artifact,
                "minority_shape_f1": minority,
            },
            "gates": {
                "shape_macro_f1": SHAPE_MACRO_F1_GATE,
                "fill_macro_f1": FILL_MACRO_F1_GATE,
                "artifact_f1": ARTIFACT_F1_GATE,
                "minority_class_f1": MINORITY_CLASS_F1_GATE,
                "packed_onnx_parity": PARITY_TOLERANCE,
            },
            "gate_results": gate_results,
            "packed_onnx_maximum_absolute_error": parity_max,
            "inference_total_ms": round(inference_ms, 3),
            "inference_ms_per_patch": round(inference_ms / len(samples), 6),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        require_evaluator_identity(
            expected_task=definition.task,
            expected_revision=definition.revision,
            manifest=manifest_payload,
            split_config=split_config,
            seal_binding=canonical_seal.binding,
            report=report,
        )
        report_path = output_dir / "public-gate-report.json"
        _write_json(report_path, report)
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        canonical_result = complete_gate_seal(canonical_seal, status=str(report["status"]), report_sha256=report_sha256)
        _write_json(
            seal_path,
            {
                "status": "completed",
                "evaluation_count": 1,
                "manifest_sha256": manifest_sha256,
                "report_sha256": report_sha256,
                "canonical_seal_key": canonical_seal.key,
                "canonical_result_path": str(canonical_result),
                "binding": canonical_seal.binding,
            },
        )
        return report
    except Exception as error:
        _write_json(seal_path, {"status": "failed_after_open", "evaluation_count": 1, "error_type": type(error).__name__})
        raise


def evaluate_public_gate(checkpoint: Path, onnx_path: Path, output_dir: Path) -> dict[str, object]:
    samples = build_public_gate_split()
    return _evaluate_gate(
        checkpoint,
        onnx_path,
        output_dir,
        definition=CLASSIFIER_PUBLIC_GATE,
        samples=samples,
        manifest_payload=public_gate_manifest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public_gate(args.checkpoint, args.onnx, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
