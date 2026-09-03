# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Evaluate fixed-radius marker geometry consensus on the V13 dev split.

This is an aggregate-only feasibility check.  It reuses the V21 P1 ONNX
payload, V13 proposals and geometry filtering, confidence threshold, offset
decoding, radius clipping, and radius-aware NMS.  The sole postprocessing
change is to test the existing ring-support/center-density rule at each fixed
integer radius from 3 through 12 and accept when any radius passes.
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
    extract_proposals,
)
from ml.markers.center.metrics import center_metrics
from ml.markers.center.proposal_geometry_v13.dataset import (
    build_selection_scenes,
    selection_manifest,
)
from ml.markers.center.proposal_geometry_v13.geometry import filter_proposals


REPO_ROOT = Path(__file__).resolve().parents[4]
V21_ARTIFACT_RELATIVE = Path(
    "artifacts/goal22-worktrees/marker-v21/ml/markers/center/"
    "focal_confidence_v21/artifacts/P1-run"
)


def _find_workspace_artifact(relative: Path) -> Path:
    for root in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = root / relative
        if candidate.exists():
            return candidate
    return REPO_ROOT / relative


def _display_path(path: Path) -> str:
    for root in (REPO_ROOT, *REPO_ROOT.parents):
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name


V21_ARTIFACT_DIR = _find_workspace_artifact(V21_ARTIFACT_RELATIVE)
DEFAULT_ONNX = V21_ARTIFACT_DIR / "marker-center-focal-confidence-v21-p1.onnx"
V21_RESULT = REPO_ROOT / "ml/markers/center/focal_confidence_v21/P1_RESULT.json"
V21_DIAGNOSTIC = REPO_ROOT / "ml/markers/center/focal_confidence_v21/diagnostics/V21_DIAGNOSTIC.json"
V22_DIAGNOSTIC = REPO_ROOT / "ml/markers/center/veto_override_v22/V22_FEASIBILITY_DIAGNOSTIC.json"
FIXED_THRESHOLD = 0.25
RING_RADII = tuple(range(3, 13))
INK_THRESHOLD = 0.12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_sha256() -> str:
    encoded = (json.dumps(selection_manifest(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ring_support(scene: Any, x: float, y: float, radius: int) -> int:
    ix, iy = int(round(x)), int(round(y))
    points = (
        (ix - radius, iy), (ix + radius, iy),
        (ix, iy - radius), (ix, iy + radius),
        (ix - radius, iy - radius), (ix + radius, iy - radius),
        (ix - radius, iy + radius), (ix + radius, iy + radius),
    )
    ink = scene.tensor[0]
    return sum(
        1
        for px, py in points
        if 0 <= px < ink.shape[1]
        and 0 <= py < ink.shape[0]
        and float(ink[py, px]) >= INK_THRESHOLD
    )


def _center_density(scene: Any, x: float, y: float) -> float:
    ix, iy = int(round(x)), int(round(y))
    ink = scene.tensor[0]
    return float(torch_mean(ink[max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3]))


def torch_mean(values: Any) -> float:
    """Small indirection keeps this module free of any changed model code."""
    return values.mean().item()


def multiradius_geometry_consensus(scene: Any, x: float, y: float) -> bool:
    """Apply the existing support>=3 or density>=0.28 rule at radii 3..12."""
    if _center_density(scene, x, y) >= 0.28:
        return True
    return any(_ring_support(scene, x, y, radius) >= 3 for radius in RING_RADII)


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


def postprocess_multiradius(
    scene: Any,
    proposals: Any,
    output: np.ndarray,
) -> tuple[MarkerPrediction, ...]:
    """Decode V21 output with only the marker geometry rule replaced."""
    if output.shape != (len(proposals.patches), 4):
        raise ValueError("V21 candidate output must be NC [candidate_count,4]")
    candidates: list[MarkerPrediction] = []
    for index in np.flatnonzero(output[:, 0] >= FIXED_THRESHOLD):
        base_x, base_y = proposals.coordinates[index].tolist()
        x = float(base_x + output[index, 1] * 4.0)
        y = float(base_y + output[index, 2] * 4.0)
        radius = float(np.clip(output[index, 3], 2.5, 8.0))
        if _center_is_unmasked(scene, x, y) and multiradius_geometry_consensus(scene, x, y):
            candidates.append(MarkerPrediction(x, y, radius, float(output[index, 0])))
    return _nms(candidates)


def _prohibited_hits(predictions: tuple[MarkerPrediction, ...], scene: Any) -> Counter[str]:
    hits: Counter[str] = Counter()
    for prediction in predictions:
        for item in scene.prohibited:
            if math.hypot(prediction.x - item.x, prediction.y - item.y) <= MATCH_TOLERANCE:
                hits[item.kind] += 1
    return hits


def summarize(onnx_path: Path = DEFAULT_ONNX) -> dict[str, Any]:
    """Run aggregate-only V23 feasibility on the fixed V13 synthetic dev split."""
    started = time.perf_counter()
    for path, label in (
        (onnx_path, "V21 ONNX"),
        (V21_RESULT, "V21 result"),
        (V21_DIAGNOSTIC, "V21 diagnostic"),
        (V22_DIAGNOSTIC, "V22 diagnostic"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} artifact is missing: {path}")

    scenes = build_selection_scenes("dev")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    truth_count = accepted_count = true_positives = false_positives = false_negatives = 0
    duplicate_count = 0
    prohibited: Counter[str] = Counter()
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
        predictions = postprocess_multiradius(scene, filtered, output)
        metrics = center_metrics(predictions, scene.centers, MATCH_TOLERANCE)
        truth_count += len(scene.centers)
        accepted_count += len(predictions)
        true_positives += metrics.true_positives
        false_positives += metrics.false_positives
        false_negatives += metrics.false_negatives
        duplicate_count += metrics.duplicate_count
        prohibited.update(_prohibited_hits(predictions, scene))

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / truth_count if truth_count else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prohibited_count = sum(prohibited.values())
    clears = precision >= 0.95 and recall >= 0.95 and prohibited_count == 0
    return {
        "schema": "graphreader.marker-center-multiradius-geometry-v23-feasibility.v1",
        "revision": "marker-center-multiradius-geometry-v23-feasibility",
        "status": "feasible_startable" if clears else "failed_feasibility_no_candidate",
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
            "v21_onnx_path": _display_path(onnx_path),
            "v21_onnx_sha256": _sha256(onnx_path),
            "v22_diagnostic_path": V22_DIAGNOSTIC.relative_to(REPO_ROOT).as_posix(),
            "v22_diagnostic_sha256": _sha256(V22_DIAGNOSTIC),
            "proposal_revision": "marker-center-proposal-geometry-v13",
        },
        "inference": {
            "provider": session.get_providers()[0],
            "scene_count": len(scenes),
            "truth_count": truth_count,
            "accepted_candidate_count": accepted_count,
            "raw_proposal_count": raw_count,
            "geometry_filtered_proposal_count": geometry_count,
            "confidence_threshold": FIXED_THRESHOLD,
            "offset_scale": 4.0,
            "radius_clip_px": [2.5, 8.0],
            "ring_radii_px": list(RING_RADII),
            "nms": "V21 unchanged radius-aware NMS",
            "artifact_mask_veto": "V21 unchanged",
            "geometry_rule": "any radius 3..12 has ring support>=3 or center density>=0.28",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        },
        "metrics": {
            "accepted_candidate_count": accepted_count,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "miss_count": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "duplicate_count": duplicate_count,
            "prohibited_structure_hits": prohibited_count,
            "prohibited_structure_hit_rate": prohibited_count / truth_count if truth_count else 0.0,
            "prohibited_hits_by_kind": dict(sorted(prohibited.items())),
        },
        "acceptance_bars": {
            "precision_minimum": 0.95,
            "recall_minimum": 0.95,
            "prohibited_hits_maximum": 0,
        },
        "feasibility": {
            "startable": clears,
            "reason": "multiradius geometry clears both 0.95 bars with zero prohibited hits"
            if clears
            else "multiradius geometry does not clear both 0.95 precision and recall bars with zero prohibited hits",
        },
        "candidate": {
            "optimizer_steps": 0,
            "ring_radii_px": list(RING_RADII),
        } if clears else None,
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
