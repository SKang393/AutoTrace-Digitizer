# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact V10, official spacing-P2, and numeric-V5 composed execution."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256

import numpy as np

from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_spaced_recall_detector_v10.dataset import encode_proposal
from ml.ocr.production_composition_v2.pipeline import (
    DirectRunner,
    DirectTensorEvidence,
    _distance,
    _numeric_recognize,
    _official_recognize,
    _role,
    _softmax,
)

from .dataset import CompositionScene, TextTruth
from .protocol import DETECTOR_THRESHOLD, NUMERIC_THRESHOLD, TRUTH_MATCH_IOU_MINIMUM


def evaluate_scenes(
    scenes: tuple[CompositionScene, ...], detector_runner: DirectRunner,
    official_runner: DirectRunner, numeric_runner: DirectRunner, official_alphabet: str,
) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    true_positives = false_positives = false_negatives = duplicates = 0
    exact = errors = characters = role_correct = changed_nonspace = 0
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    forbidden_numeric_routes = spacing_changes = 0
    route_counts: dict[str, int] = defaultdict(int)
    for scene in scenes:
        candidates = proposals(scene.raster)
        values = np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)
        probabilities = _softmax(detector_runner.run(values))[:, 1]
        accepted = [
            item for item, probability in zip(candidates, probabilities, strict=True)
            if probability >= DETECTOR_THRESHOLD
        ]
        matched_truths: set[int] = set()
        scene_false_positives = scene_duplicates = 0
        predictions: list[dict[str, object]] = []
        for candidate in accepted:
            matches = [
                index for index, truth in enumerate(scene.truths)
                if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM
            ]
            if not matches:
                scene_false_positives += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box))
            if best in matched_truths:
                scene_duplicates += 1
                continue
            matched_truths.add(best)
            truth: TextTruth = scene.truths[best]
            raw, official_text = _official_recognize(scene.raster, candidate.box, official_runner, official_alphabet)
            numeric_text, numeric_confidence = _numeric_recognize(scene.raster, candidate.box, numeric_runner)
            official_role, numeric_role = _role(candidate.box, official_text), _role(candidate.box, numeric_text)
            select_numeric = bool(numeric_text) and numeric_confidence >= NUMERIC_THRESHOLD and numeric_role in {
                "x_tick", "y_tick",
            }
            prediction, predicted_role, route = (
                (numeric_text, numeric_role, "numeric_specialist") if select_numeric
                else (official_text, official_role, "general_recognizer")
            )
            matched = prediction == truth.truth_text
            route_counts[route] += 1
            forbidden_numeric_routes += int(select_numeric and truth.role not in {"x_tick", "y_tick"})
            exact += int(matched)
            errors += _distance(truth.truth_text, prediction)
            characters += len(truth.truth_text)
            role_correct += int(predicted_role == truth.role)
            family_counts[truth.family][0] += int(matched)
            family_counts[truth.family][1] += 1
            spacing_changes += int(official_text != raw)
            changed_nonspace += int(official_text != raw and " " not in truth.truth_text)
            predictions.append({
                "truth_text": truth.truth_text, "display_text": truth.display_text, "truth_role": truth.role,
                "text_family": truth.family, "prediction": prediction, "predicted_role": predicted_role,
                "route": route, "official_raw_prediction": raw, "official_prediction": official_text,
                "numeric_prediction": numeric_text, "numeric_confidence": numeric_confidence,
                "proposal_bbox": [candidate.box.left, candidate.box.top, candidate.box.right, candidate.box.bottom],
                "truth_bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "exact": matched,
            })
        scene_false_negatives = len(scene.truths) - len(matched_truths)
        true_positives += len(matched_truths)
        false_positives += scene_false_positives
        false_negatives += scene_false_negatives
        duplicates += scene_duplicates
        cases.append({
            "scene_id": scene.scene_id,
            "source_raster_sha256": sha256(scene.raster.tobytes(order="C")).hexdigest(),
            "truth_region_count": len(scene.truths), "proposal_count": len(candidates),
            "accepted_region_count": len(accepted), "true_positives": len(matched_truths),
            "false_positives": scene_false_positives, "false_negatives": scene_false_negatives,
            "duplicate_region_count": scene_duplicates, "prohibited_structure_hits": scene_false_positives,
            "exact_detection": scene_false_positives == scene_false_negatives == scene_duplicates == 0,
            "predictions": predictions,
        })
    total = sum(len(scene.truths) for scene in scenes)
    return {
        "scene_count": len(scenes), "truth_region_count": total,
        "exact_detection_scene_count": sum(int(item["exact_detection"]) for item in cases),
        "true_positives": true_positives, "false_positives": false_positives,
        "false_negatives": false_negatives, "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_positives, "recognition_exact_match": exact / max(1, total),
        "character_error_rate": errors / max(1, characters), "role_accuracy": role_correct / max(1, total),
        "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
        "word_exact_match": family_counts["word"][0] / max(1, family_counts["word"][1]),
        "ambiguity_exact_match": family_counts["ambiguity"][0] / max(1, family_counts["ambiguity"][1]),
        "spacing_changed_count": spacing_changes,
        "spacing_changed_nonspace_truth_count": changed_nonspace,
        "forbidden_numeric_route_count": forbidden_numeric_routes,
        "route_counts": dict(sorted(route_counts.items())), "marker_creation_evaluated": False, "cases": cases,
    }


__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
