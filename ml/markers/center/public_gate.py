# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use public gate for the frozen marker-center ONNX candidate."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    require_evaluator_identity,
)
from .dataset import ARTIFACT_KINDS, _artifact_geometry, _draw_marker
from .metrics import center_metrics
from .postprocess import detect_heads


PUBLIC_GATE_REVISION = "marker-center-public-gate-v1"
PUBLIC_GATE_SEED = 20260804
PUBLIC_GATE_CONFIG_PATH = Path("ml/markers/center/gates/public-v1.json")
PUBLIC_GATE_CONFIG = {
    "exact_count_every_fixture": True,
    "maximum_duplicate_count": 0,
    "maximum_prohibited_structure_hits": 0,
    "matching_tolerance_px": 5.0,
    "prohibited_structure_kinds": list(ARTIFACT_KINDS),
}
MARKER_CENTER_TASK = "marker-center"


@dataclass(frozen=True)
class CenterGateDefinition:
    task: str
    revision: str
    split_config_path: Path
    evaluator_source_paths: tuple[Path, ...]


CENTER_PUBLIC_GATE = CenterGateDefinition(
    task=MARKER_CENTER_TASK,
    revision=PUBLIC_GATE_REVISION,
    split_config_path=PUBLIC_GATE_CONFIG_PATH,
    evaluator_source_paths=(
        Path("ml/markers/center/public_gate.py"),
        Path("ml/markers/gate_seal.py"),
        Path("ml/markers/center/dataset.py"),
        Path("ml/markers/center/metrics.py"),
        Path("ml/markers/center/postprocess.py"),
    ),
)


@dataclass(frozen=True)
class PublicGateScene:
    scene_id: str
    family: str
    template: str
    tensor: np.ndarray
    centers: tuple[tuple[float, float], ...]
    hard_negatives: tuple[tuple[str, float, float], ...]


def _degrade(image: Image.Image, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "fax_resample":
        image = image.resize((112, 112), Image.Resampling.BILINEAR).resize((128, 128), Image.Resampling.BICUBIC)
    elif family == "toner_gamma":
        image = image.filter(ImageFilter.GaussianBlur(0.22))
    elif family == "vertical_banding":
        image = image.filter(ImageFilter.MedianFilter(3))
    else:
        raise ValueError(family)
    luminance = np.asarray(image, dtype=np.float32) / 255.0
    if family == "toner_gamma":
        luminance = np.power(luminance, 1.08)
    elif family == "vertical_banding":
        luminance[:, ::11] = np.clip(luminance[:, ::11] - 0.025, 0.0, 1.0)
    luminance = np.clip(luminance + rng.normal(0.0, 0.0025, luminance.shape), 0.0, 1.0)
    return luminance.astype(np.float32)


def _render_scene(
    scene_id: str,
    family: str,
    template: str,
    centers: tuple[tuple[int, int], ...],
    seed: int,
) -> PublicGateScene:
    image = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(image)
    text_image = Image.new("L", image.size, 0)
    text_draw = ImageDraw.Draw(text_image)
    artifact_image = Image.new("L", image.size, 0)
    artifact_draw = ImageDraw.Draw(artifact_image)
    for first, second in zip(centers, centers[1:]):
        draw.line((*first, *second), fill=42, width=1)
    for index, center in enumerate(centers):
        _draw_marker(draw, center, 3 + ((index + seed) % 3), index + seed)

    locations = ((14, 105), (29, 105), (44, 104), (59, 105), (75, 104), (91, 105), (107, 104), (112, 16))
    hard_negatives: list[tuple[str, float, float]] = []
    for kind, (x, y) in zip(ARTIFACT_KINDS, locations, strict=True):
        target = text_draw if kind == "text" else artifact_draw
        _artifact_geometry(draw, target, kind, x, y)
        hard_negatives.append((kind, float(x), float(y)))

    luminance = _degrade(image, family, seed)
    tensor = np.stack(
        (
            1.0 - luminance,
            np.asarray(text_image, dtype=np.float32) / 255.0,
            np.asarray(artifact_image, dtype=np.float32) / 255.0,
        ),
        axis=0,
    ).astype(np.float32)
    return PublicGateScene(
        scene_id,
        family,
        template,
        tensor,
        tuple((float(x), float(y)) for x, y in centers),
        tuple(hard_negatives),
    )


def build_public_gate_split() -> tuple[PublicGateScene, ...]:
    definitions = (
        ("public-zigzag", "fax_resample", "alternating_zigzag", ((17, 26), (34, 51), (51, 33), (68, 68), (85, 42), (103, 77))),
        ("public-probes", "toner_gamma", "paired_probe_columns", ((21, 67), (39, 44), (57, 70), (75, 37), (94, 58), (109, 31))),
        ("public-stair", "vertical_banding", "descending_stair", ((18, 75), (35, 66), (52, 58), (69, 49), (86, 40), (103, 31))),
    )
    return tuple(
        _render_scene(scene_id, family, template, centers, PUBLIC_GATE_SEED + index)
        for index, (scene_id, family, template, centers) in enumerate(definitions)
    )


def public_gate_manifest() -> dict[str, object]:
    cases = []
    for scene in build_public_gate_split():
        cases.append(
            {
                "scene_id": scene.scene_id,
                "family": scene.family,
                "template": scene.template,
                "tensor_sha256": hashlib.sha256(scene.tensor.tobytes(order="C")).hexdigest(),
                "center_count": len(scene.centers),
                "centers": scene.centers,
                "hard_negative_kinds": [item[0] for item in scene.hard_negatives],
            }
        )
    return {
        "manifest_version": 1,
        "task": MARKER_CENTER_TASK,
        "revision": PUBLIC_GATE_REVISION,
        "seed": PUBLIC_GATE_SEED,
        "selection_use": False,
        "private_data": False,
        "cases": cases,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value))


def center_gate_results(
    per_scene: list[dict[str, object]],
    total_duplicates: int,
    aggregate_hits: dict[str, int],
) -> dict[str, bool]:
    return {
        "exact_count_every_fixture": all(bool(row["exact_count"]) for row in per_scene),
        "zero_duplicates": total_duplicates == 0,
        "zero_prohibited_structure_hits": not any(aggregate_hits.values()),
    }


def _evaluate_gate(
    onnx_path: Path,
    output_dir: Path,
    *,
    definition: CenterGateDefinition,
    scenes: tuple[PublicGateScene, ...],
    manifest_payload: dict[str, object],
) -> dict[str, object]:
    selected_scenes = scenes
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
    candidate_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    canonical_seal = acquire_gate_seal(
        repo_root=repo_root,
        task=definition.task,
        revision=definition.revision,
        candidate_hashes={"onnx_sha256": candidate_sha256},
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
        session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        per_scene = []
        inference_ms = 0.0
        total_duplicates = 0
        aggregate_hits = {kind: 0 for kind in ARTIFACT_KINDS}
        for scene in selected_scenes:
            tensor = scene.tensor[None].copy()
            inference_started = time.perf_counter()
            heads = session.run([output_name], {input_name: tensor})[0]
            inference_ms += (time.perf_counter() - inference_started) * 1000.0
            detections = detect_heads(
                heads,
                text_mask=tensor[0, 1],
                artifact_mask=tensor[0, 2],
            )
            metrics = center_metrics(detections, scene.centers, 5.0)
            total_duplicates += metrics.duplicate_count
            hits = {kind: 0 for kind in ARTIFACT_KINDS}
            for kind, x, y in scene.hard_negatives:
                hits[kind] += sum((item.x - x) ** 2 + (item.y - y) ** 2 <= 8.0**2 for item in detections)
                aggregate_hits[kind] += hits[kind]
            exact = metrics.false_positives == 0 and metrics.false_negatives == 0 and metrics.duplicate_count == 0
            per_scene.append(
                {
                    "scene_id": scene.scene_id,
                    "truth_count": len(scene.centers),
                    "prediction_count": len(detections),
                    "metrics_5px": metrics.to_dict(),
                    "hard_negative_hits": hits,
                    "exact_count": exact,
                }
            )
        gate_results = center_gate_results(per_scene, total_duplicates, aggregate_hits)
        report = {
            "status": "pass" if all(gate_results.values()) else "fail",
            "release_eligible": False,
            "release_blocker": "Production runtime discovery and packaging evidence are outside this marker-owned gate.",
            "task": definition.task,
            "revision": definition.revision,
            "evaluation_count": 1,
            "dataset_manifest_sha256": manifest_sha256,
            "onnx_sha256": candidate_sha256,
            "canonical_seal_key": canonical_seal.key,
            "seal_binding": canonical_seal.binding,
            "provider": session.get_providers()[0],
            "scene_count": len(per_scene),
            "gate_results": gate_results,
            "hard_negative_hits": aggregate_hits,
            "per_scene": per_scene,
            "inference_total_ms": round(inference_ms, 3),
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


def evaluate_public_gate(onnx_path: Path, output_dir: Path) -> dict[str, object]:
    scenes = build_public_gate_split()
    return _evaluate_gate(
        onnx_path,
        output_dir,
        definition=CENTER_PUBLIC_GATE,
        scenes=scenes,
        manifest_payload=public_gate_manifest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public_gate(args.onnx, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
