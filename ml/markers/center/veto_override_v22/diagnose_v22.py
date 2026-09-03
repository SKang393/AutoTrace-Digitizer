# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Evaluate a bounded high-confidence veto override on the fixed V13 dev split.

This module never trains, opens a public or sealed split, or emits case-level
identities, predictions, truth rows, or pixels. It reuses the V21 ONNX output
and V13 proposals, offsets, radii, and NMS exactly, changing only the
postprocessing decision for candidates rejected by the artifact or marker
geometry vetoes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort

from ml.markers.center.line_aware_v1.pipeline import (
    MATCH_TOLERANCE,
    MarkerPrediction,
    _center_is_unmasked,
    _marker_geometry_consensus,
    extract_proposals,
)
from ml.markers.center.metrics import center_metrics
from ml.markers.center.proposal_geometry_v13.dataset import (
    build_selection_scenes,
    selection_manifest,
)
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals


REPO_ROOT = Path(__file__).resolve().parents[4]
V21_ARTIFACT_DIR = (
    REPO_ROOT
    / "artifacts/goal22-worktrees/marker-v21/ml/markers/center/"
    / "focal_confidence_v21/artifacts/P1-run"
)
DEFAULT_ONNX = V21_ARTIFACT_DIR / "marker-center-focal-confidence-v21-p1.onnx"
V21_RESULT = REPO_ROOT / "ml/markers/center/focal_confidence_v21/P1_RESULT.json"
V21_DIAGNOSTIC = REPO_ROOT / "ml/markers/center/focal_confidence_v21/diagnostics/V21_DIAGNOSTIC.json"
FIXED_THRESHOLD = 0.25
OVERRIDE_FLOORS = (0.90, 0.95, 0.99, 0.995, 0.999)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256() -> str:
    encoded = (json.dumps(selection_manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _nms(candidates: list[MarkerPrediction]) -> tuple[MarkerPrediction, ...]:
    accepted: list[MarkerPrediction] = []
    for candidate in sorted(candidates, key=lambda item: (-item.confidence, item.y, item.x)):
        if any(
            math.hypot(candidate.x - current.x, candidate.y - current.y)
            < max(5.0, 1.25 * max(candidate.radius, current.radius))
            for current in accepted
        ):
            continue
        accepted.append(candidate)
    return tuple(sorted(accepted, key=lambda item: (item.y, item.x, -item.confidence)))


def postprocess_with_veto_override(
    scene: Any,
    proposals: Any,
    output: np.ndarray,
    *,
    override_floor: float,
) -> tuple[tuple[MarkerPrediction, ...], int, int]:
    """Apply V21 postprocessing with only a fixed high-confidence veto bypass.

    The first returned count is every candidate admitted by the override
    before NMS. The second is the count remaining after the unchanged NMS.
    """
    if output.shape != (len(proposals.patches), 4):
        raise ValueError("V21 candidate output must be NC [candidate_count,4]")
    candidates: list[MarkerPrediction] = []
    bypassed = 0
    for index in np.flatnonzero(output[:, 0] >= FIXED_THRESHOLD):
        base_x, base_y = proposals.coordinates[index].tolist()
        x = float(base_x + output[index, 1] * 4.0)
        y = float(base_y + output[index, 2] * 4.0)
        radius = float(np.clip(output[index, 3], 2.5, 8.0))
        unmasked = _center_is_unmasked(scene, x, y)
        geometric = _marker_geometry_consensus(scene, x, y, radius)
        if unmasked and geometric:
            candidates.append(MarkerPrediction(x, y, radius, float(output[index, 0])))
            continue
        if float(output[index, 0]) >= override_floor:
            candidates.append(MarkerPrediction(x, y, radius, float(output[index, 0])))
            bypassed += 1
    accepted = _nms(candidates)
    return accepted, bypassed, sum(
        1
        for prediction in accepted
        if not (_center_is_unmasked(scene, prediction.x, prediction.y)
                and _marker_geometry_consensus(scene, prediction.x, prediction.y, prediction.radius))
    )


def _prohibited_hits(predictions: tuple[MarkerPrediction, ...], scene: Any) -> Counter[str]:
    hits: Counter[str] = Counter()
    for prediction in predictions:
        for item in scene.prohibited:
            if math.hypot(prediction.x - item.x, prediction.y - item.y) <= MATCH_TOLERANCE:
                hits[item.kind] += 1
    return hits


def _floor_result(
    scenes: tuple[Any, ...],
    outputs: tuple[tuple[Any, np.ndarray], ...],
    override_floor: float,
) -> dict[str, Any]:
    metrics = []
    prohibited: Counter[str] = Counter()
    bypassed = 0
    bypassed_after_nms = 0
    for scene, (proposals, output) in zip(scenes, outputs, strict=True):
        predictions, admitted, accepted_bypasses = postprocess_with_veto_override(
            scene, proposals, output, override_floor=override_floor
        )
        metrics.append(center_metrics(predictions, scene.centers, MATCH_TOLERANCE))
        prohibited.update(_prohibited_hits(predictions, scene))
        bypassed += admitted
        bypassed_after_nms += accepted_bypasses
    truth_count = sum(len(scene.centers) for scene in scenes)
    true_positives = sum(item.true_positives for item in metrics)
    false_positives = sum(item.false_positives for item in metrics)
    false_negatives = sum(item.false_negatives for item in metrics)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / truth_count if truth_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prohibited_count = sum(prohibited.values())
    return {
        "override_floor": override_floor,
        "center_threshold": FIXED_THRESHOLD,
        "match_tolerance_px": MATCH_TOLERANCE,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "duplicate_count": sum(item.duplicate_count for item in metrics),
        "prohibited_structure_hits": prohibited_count,
        "prohibited_structure_hit_rate": prohibited_count / truth_count if truth_count else 0.0,
        "prohibited_hits_by_kind": dict(sorted(prohibited.items())),
        "bypassed_candidates": bypassed,
        "bypassed_candidates_after_nms": bypassed_after_nms,
        "acceptance_bars": {"precision_minimum": 0.95, "recall_minimum": 0.95, "prohibited_hits_maximum": 0},
        "clears_both_bars": precision >= 0.95 and recall >= 0.95 and prohibited_count == 0,
    }


def summarize(
    onnx_path: Path = DEFAULT_ONNX,
    *,
    floors: tuple[float, ...] = OVERRIDE_FLOORS,
) -> dict[str, Any]:
    """Run aggregate-only V22 feasibility on V13 synthetic dev bytes."""
    started = time.perf_counter()
    for path, label in ((onnx_path, "V21 ONNX"), (V21_RESULT, "V21 result"), (V21_DIAGNOSTIC, "V21 diagnostic")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")
    scenes = build_selection_scenes("dev")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs: list[tuple[Any, np.ndarray]] = []
    raw_count = geometry_count = 0
    for scene in scenes:
        raw = extract_proposals(scene.tensor)
        filtered = filter_proposals(scene.tensor, raw)
        raw_count += len(raw.coordinates)
        geometry_count += len(filtered.coordinates)
        output = session.run(
            [output_name],
            {input_name: filtered.patches.numpy().astype(np.float32, copy=False)},
        )[0]
        outputs.append((filtered, output))
    floor_results = {str(floor): _floor_result(scenes, tuple(outputs), floor) for floor in floors}
    passing = [float(floor) for floor, result in floor_results.items() if result["clears_both_bars"]]
    return {
        "schema": "graphreader.marker-center-veto-override-v22-feasibility.v1",
        "revision": "marker-center-veto-override-v22-feasibility",
        "status": "feasible_startable" if passing else "failed_feasibility_no_candidate",
        "scope": {
            "synthetic_only": True,
            "fixed_dev_split": "marker-center-proposal-geometry-v13-dev",
            "fixed_dev_manifest_sha256": _manifest_sha256(),
            "private_or_article_images": False,
            "public_gate_archive_opened": False,
            "sealed_runs": 0,
            "scene_ids_emitted": False,
            "case_level_details_emitted": False,
            "training_performed": False,
            "optimizer_steps": 0,
        },
        "binding": {
            "v21_result_path": V21_RESULT.relative_to(REPO_ROOT).as_posix(),
            "v21_result_sha256": _sha256(V21_RESULT),
            "v21_diagnostic_path": V21_DIAGNOSTIC.relative_to(REPO_ROOT).as_posix(),
            "v21_diagnostic_sha256": _sha256(V21_DIAGNOSTIC),
            "v21_onnx_sha256": _sha256(onnx_path),
            "proposal_revision": "marker-center-proposal-geometry-v13",
        },
        "inference": {
            "provider": session.get_providers()[0],
            "truth_count": sum(len(scene.centers) for scene in scenes),
            "scene_count": len(scenes),
            "raw_proposal_count": raw_count,
            "geometry_filtered_proposal_count": geometry_count,
            "retained_threshold": FIXED_THRESHOLD,
            "offset_scale": 4.0,
            "radius_clip_px": [2.5, 8.0],
            "nms": "V21 unchanged radius-aware NMS",
            "ordinary_vetoes": ["artifact-mask", "marker-geometry"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "override_floors": floor_results,
        "passing_floors": passing,
        "feasibility": {
            "startable": bool(passing),
            "reason": "at least one floor clears precision and recall with zero prohibited hits"
            if passing else "no swept floor clears both 0.95 precision and recall bars with zero prohibited hits",
        },
        "candidate": None if not passing else {"override_floor": min(passing), "optimizer_steps": 0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.onnx.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
