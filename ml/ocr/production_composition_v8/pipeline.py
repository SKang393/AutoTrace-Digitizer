# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""V6 composition with bounded zero consensus and exposed ambiguity aliases."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import re
from typing import Any

import numpy as np

from ml.ocr.ambiguity_source_group_classifier_v3.crop import active_groups, group_tensor
from ml.ocr.ambiguity_source_group_classifier_v3.protocol import GLYPHS as AMBIGUITY_GLYPHS
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_spaced_recall_detector_v10.dataset import encode_proposal
from ml.ocr.official_bakeoff.production_evaluate import decode_ctc, recognition_tensor
from ml.ocr.official_recognition_spacing_v3.spacing import restore_conservative_source_spaces
from ml.ocr.production_composition_v2.pipeline import (
    DirectRunner, DirectTensorEvidence, _distance, _numeric_recognize, _role,
    _softmax, _source_crop,
)
from ml.ocr.production_composition_v6.pipeline import _force_expected_ambiguity_groups
from .dataset import CompositionScene
from .protocol import (
    AMBIGUITY_INPUT_ALIASES, CONSENSUS_RESCUE_SCORE_MINIMUM, DETECTOR_THRESHOLD,
    NUMERIC_THRESHOLD, OFFICIAL_RESCUE_SCORE_MINIMUM, TRUTH_MATCH_IOU_MINIMUM,
    ZERO_CONSENSUS_RESCUE_SCORE_MINIMUM,
)


_GRAPH_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?%?$")
_EXTENDED_AMBIGUITY = frozenset((*AMBIGUITY_GLYPHS, *AMBIGUITY_INPUT_ALIASES))


def _official_recognize(gray: np.ndarray, box: Any, official: DirectRunner,
                        ambiguity: DirectRunner, alphabet: str) -> tuple[str, str, str, int]:
    crop = _source_crop(gray, box, 8, 2)
    raw = decode_ctc(official.run(recognition_tensor(crop)), alphabet)
    conservative = restore_conservative_source_spaces(crop, raw)
    if _GRAPH_NUMBER.fullmatch(raw.strip()) and not _GRAPH_NUMBER.fullmatch(conservative.strip()):
        conservative = raw
    final = conservative
    nonspace = [character for character in conservative if not character.isspace()]
    all_extended = bool(nonspace) and all(character in _EXTENDED_AMBIGUITY for character in nonspace)
    groups = active_groups(crop)
    if all_extended:
        groups = _force_expected_ambiguity_groups(crop, len(nonspace))
    changed = 0
    if len(groups) == len(nonspace) and any(character in _EXTENDED_AMBIGUITY for character in nonspace):
        indices = list(range(len(nonspace))) if all_extended else [
            index for index, character in enumerate(nonspace) if character in AMBIGUITY_GLYPHS
        ]
        values = np.stack([group_tensor(crop, groups, index) for index in indices]).astype(np.float32)
        logits = ambiguity.run(values)
        for index, label in zip(indices, np.argmax(logits, axis=1), strict=True):
            replacement = AMBIGUITY_GLYPHS[int(label)]
            changed += int(nonspace[index] != replacement)
            nonspace[index] = replacement
        if all(character in AMBIGUITY_GLYPHS for character in nonspace):
            final = " ".join(nonspace)
        else:
            iterator = iter(nonspace)
            final = "".join(character if character.isspace() else next(iterator) for character in conservative)
    return raw, conservative, final, changed


def evaluate_scenes(scenes: tuple[CompositionScene, ...], detector: DirectRunner, official: DirectRunner,
                    numeric: DirectRunner, ambiguity: DirectRunner, alphabet: str) -> dict[str, object]:
    cases = []
    totals = defaultdict(int)
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    routes = defaultdict(int)
    for scene in scenes:
        candidates = proposals(scene.raster)
        probabilities = _softmax(
            detector.run(np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32))
        )[:, 1]
        accepted: list[tuple[int, Any, str, tuple[str, str, str, int] | None, tuple[str, float] | None]] = []
        for candidate_index, (candidate, probability) in enumerate(zip(candidates, probabilities, strict=True)):
            if probability >= DETECTOR_THRESHOLD:
                accepted.append((candidate_index, candidate, "detector", None, None))
                continue
            if probability < ZERO_CONSENSUS_RESCUE_SCORE_MINIMUM:
                continue
            official_result = _official_recognize(scene.raster, candidate.box, official, ambiguity, alphabet)
            official_text = official_result[2].strip()
            official_role = _role(candidate.box, official_text)
            if not (_GRAPH_NUMBER.fullmatch(official_text) and official_role in {"x_tick", "y_tick"}):
                continue
            if probability >= OFFICIAL_RESCUE_SCORE_MINIMUM:
                accepted.append((candidate_index, candidate, "official_tick_rescue", official_result, None))
                continue
            numeric_result = _numeric_recognize(scene.raster, candidate.box, numeric)
            numeric_text, confidence = numeric_result
            numeric_role = _role(candidate.box, numeric_text)
            exact_consensus = (
                numeric_text == official_text
                and confidence >= NUMERIC_THRESHOLD
                and numeric_role == official_role
                and numeric_role in {"x_tick", "y_tick"}
            )
            if probability >= CONSENSUS_RESCUE_SCORE_MINIMUM and exact_consensus:
                accepted.append((candidate_index, candidate, "official_numeric_consensus_rescue", official_result, numeric_result))
            elif exact_consensus and numeric_text == "0":
                accepted.append((candidate_index, candidate, "zero_numeric_consensus_rescue", official_result, numeric_result))
        matched_truths = set()
        predictions = []
        scene_fp = scene_dup = 0
        for _, candidate, acceptance_route, precomputed_official, precomputed_numeric in accepted:
            matches = [index for index, truth in enumerate(scene.truths) if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches:
                scene_fp += 1
                continue
            best = max(matches, key=lambda index: box_iou(candidate.box, scene.truths[index].box))
            if best in matched_truths:
                scene_dup += 1
                continue
            matched_truths.add(best)
            truth = scene.truths[best]
            raw, conservative_text, official_text, ambiguity_changes = (
                precomputed_official if precomputed_official is not None
                else _official_recognize(scene.raster, candidate.box, official, ambiguity, alphabet)
            )
            numeric_text, confidence = (
                precomputed_numeric if precomputed_numeric is not None
                else _numeric_recognize(scene.raster, candidate.box, numeric)
            )
            numeric_role = _role(candidate.box, numeric_text)
            official_role = _role(candidate.box, official_text)
            select_numeric = bool(numeric_text) and confidence >= NUMERIC_THRESHOLD and numeric_role in {"x_tick", "y_tick"}
            prediction, predicted_role, route = (
                (numeric_text, numeric_role, "numeric_specialist")
                if select_numeric else (official_text, official_role, "general_recognizer")
            )
            exact = prediction == truth.truth_text
            routes[route] += 1
            totals["exact"] += int(exact)
            totals["errors"] += _distance(truth.truth_text, prediction)
            totals["characters"] += len(truth.truth_text)
            totals["role_correct"] += int(predicted_role == truth.role)
            totals["forbidden_numeric"] += int(select_numeric and truth.role not in {"x_tick", "y_tick"})
            totals["forbidden_official_rescue"] += int(acceptance_route == "official_tick_rescue" and truth.role not in {"x_tick", "y_tick"})
            totals["forbidden_consensus_rescue"] += int(acceptance_route == "official_numeric_consensus_rescue" and truth.role not in {"x_tick", "y_tick"})
            totals["forbidden_zero_consensus_rescue"] += int(acceptance_route == "zero_numeric_consensus_rescue" and truth.role not in {"x_tick", "y_tick"})
            totals["spacing_changes"] += int(conservative_text != raw)
            totals["changed_nonspace"] += int(conservative_text != raw and " " not in truth.truth_text)
            totals["ambiguity_changes"] += ambiguity_changes
            totals["official_rescues"] += int(acceptance_route == "official_tick_rescue")
            totals["consensus_rescues"] += int(acceptance_route == "official_numeric_consensus_rescue")
            totals["zero_consensus_rescues"] += int(acceptance_route == "zero_numeric_consensus_rescue")
            family_counts[truth.family][0] += int(exact)
            family_counts[truth.family][1] += 1
            predictions.append({
                "truth_text": truth.truth_text, "truth_role": truth.role, "text_family": truth.family,
                "prediction": prediction, "predicted_role": predicted_role, "route": route,
                "acceptance_route": acceptance_route, "official_raw_prediction": raw,
                "official_conservative_spacing_prediction": conservative_text, "official_prediction": official_text,
                "numeric_prediction": numeric_text, "numeric_confidence": confidence,
                "ambiguity_changed_count": ambiguity_changes,
                "proposal_bbox": [candidate.box.left, candidate.box.top, candidate.box.right, candidate.box.bottom],
                "truth_bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "exact": exact,
            })
        scene_fn = len(scene.truths) - len(matched_truths)
        totals["tp"] += len(matched_truths); totals["fp"] += scene_fp; totals["fn"] += scene_fn; totals["dup"] += scene_dup
        cases.append({
            "scene_id": scene.scene_id, "source_raster_sha256": sha256(scene.raster.tobytes()).hexdigest(),
            "truth_region_count": len(scene.truths), "proposal_count": len(candidates), "accepted_region_count": len(accepted),
            "true_positives": len(matched_truths), "false_positives": scene_fp, "false_negatives": scene_fn,
            "duplicate_region_count": scene_dup, "prohibited_structure_hits": scene_fp,
            "exact_detection": scene_fp == scene_fn == scene_dup == 0, "predictions": predictions,
        })
    total = sum(len(scene.truths) for scene in scenes)
    return {
        "scene_count": len(scenes), "truth_region_count": total,
        "exact_detection_scene_count": sum(int(case["exact_detection"]) for case in cases),
        "true_positives": totals["tp"], "false_positives": totals["fp"], "false_negatives": totals["fn"],
        "duplicate_region_count": totals["dup"], "prohibited_structure_hits": totals["fp"],
        "recognition_exact_match": totals["exact"] / max(1, total),
        "character_error_rate": totals["errors"] / max(1, totals["characters"]),
        "role_accuracy": totals["role_correct"] / max(1, total),
        "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
        "word_exact_match": family_counts["word"][0] / max(1, family_counts["word"][1]),
        "ambiguity_exact_match": family_counts["ambiguity"][0] / max(1, family_counts["ambiguity"][1]),
        "official_tick_rescue_count": totals["official_rescues"],
        "official_numeric_consensus_rescue_count": totals["consensus_rescues"],
        "zero_numeric_consensus_rescue_count": totals["zero_consensus_rescues"],
        "ambiguity_changed_count": totals["ambiguity_changes"], "spacing_changed_count": totals["spacing_changes"],
        "spacing_changed_nonspace_truth_count": totals["changed_nonspace"],
        "forbidden_numeric_route_count": totals["forbidden_numeric"],
        "forbidden_official_rescue_route_count": totals["forbidden_official_rescue"],
        "forbidden_consensus_rescue_route_count": totals["forbidden_consensus_rescue"],
        "forbidden_zero_consensus_rescue_route_count": totals["forbidden_zero_consensus_rescue"],
        "route_counts": dict(sorted(routes.items())), "marker_creation_evaluated": False, "cases": cases,
    }


__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
