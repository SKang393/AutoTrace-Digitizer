# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Exact V10 plus numeric rescue, conservative spacing, and context ambiguity execution."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
from typing import Any

import numpy as np
from PIL import Image

from ml.ocr.ambiguity_context_classifier_v2.protocol import GLYPHS as AMBIGUITY_GLYPHS, IMAGE_SIZE
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals
from ml.ocr.component_spaced_recall_detector_v10.dataset import encode_proposal
from ml.ocr.official_bakeoff.production_evaluate import decode_ctc, recognition_tensor
from ml.ocr.official_recognition_spacing_v3.spacing import restore_conservative_source_spaces
from ml.ocr.production_composition_v2.pipeline import DirectRunner, DirectTensorEvidence, _distance, _numeric_recognize, _role, _softmax, _source_crop
from .dataset import CompositionScene, TextTruth
from .protocol import DETECTOR_THRESHOLD, NUMERIC_THRESHOLD, TRUTH_MATCH_IOU_MINIMUM


def _active_groups(crop: Image.Image) -> tuple[tuple[int, int, int, int], ...]:
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(edge)); contrast = max(0.0, background - float(np.percentile(gray, 1)))
    foreground = gray <= background - max(10.0, contrast * 0.30)
    rows, columns = np.where(foreground)
    if not len(columns): return ()
    active = np.flatnonzero(foreground.any(axis=0)); ink_height = int(rows.max() - rows.min() + 1)
    gap = max(5, int(math.ceil(ink_height * 0.40)))
    starts, ends = [int(active[0])], []
    for prior, current in zip(active[:-1], active[1:], strict=True):
        if int(current - prior - 1) >= gap: ends.append(int(prior)); starts.append(int(current))
    ends.append(int(active[-1]))
    result = []
    for left, right in zip(starts, ends, strict=True):
        group_rows = np.where(foreground[:, left:right + 1].any(axis=1))[0]
        result.append((left, int(group_rows[0]), right + 1, int(group_rows[-1]) + 1))
    return tuple(result)


def _ambiguity_tensor(crop: Image.Image, bounds: tuple[int, int, int, int], line_ink_height: int, baseline: int) -> np.ndarray:
    left, top, right, bottom = bounds
    gray = crop.convert("L").crop((left, top, right, bottom))
    # The tallest source group defines the common line scale. Shorter lowercase
    # groups retain their relative height and every group shares one baseline.
    scale = 11.5 / max(1, line_ink_height)
    glyph = gray.resize((max(1, round(gray.width * scale)), max(1, round(gray.height * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 255)
    paste_x = (IMAGE_SIZE - glyph.width) // 2
    paste_y = int(round(21 - (baseline - top) * scale))
    canvas.paste(glyph, (paste_x, paste_y))
    return (1.0 - np.asarray(canvas, dtype=np.float32) / 255.0)[None, :, :]


def _official_recognize(gray: np.ndarray, box: Any, official: DirectRunner, ambiguity: DirectRunner, alphabet: str) -> tuple[str, str, str, int]:
    crop = _source_crop(gray, box, 8, 2)
    raw = decode_ctc(official.run(recognition_tensor(crop)), alphabet)
    conservative = restore_conservative_source_spaces(crop, raw)
    final = conservative
    nonspace = [character for character in conservative if not character.isspace()]
    groups = _active_groups(crop)
    changed = 0
    if len(groups) == len(nonspace) and any(character in AMBIGUITY_GLYPHS for character in nonspace):
        indices = [index for index, character in enumerate(nonspace) if character in AMBIGUITY_GLYPHS]
        line_ink_height = max(group[3] - group[1] for group in groups)
        baseline = max(group[3] for group in groups)
        values = np.stack([_ambiguity_tensor(crop, groups[index], line_ink_height, baseline) for index in indices]).astype(np.float32)
        logits = ambiguity.run(values)
        for index, label in zip(indices, np.argmax(logits, axis=1), strict=True):
            replacement = AMBIGUITY_GLYPHS[int(label)]
            changed += int(nonspace[index] != replacement)
            nonspace[index] = replacement
        iterator = iter(nonspace)
        final = "".join(character if character.isspace() else next(iterator) for character in conservative)
    return raw, conservative, final, changed


def evaluate_scenes(scenes: tuple[CompositionScene, ...], detector: DirectRunner, official: DirectRunner,
                    numeric: DirectRunner, ambiguity: DirectRunner, alphabet: str) -> dict[str, object]:
    cases = []; totals = defaultdict(int); family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0]); routes = defaultdict(int)
    for scene in scenes:
        candidates = proposals(scene.raster)
        probabilities = _softmax(detector.run(np.stack([encode_proposal(scene.raster, item) for item in candidates]).astype(np.float32)))[:, 1]
        accepted = []
        rescue_records: dict[int, tuple[str, float, str]] = {}
        for candidate_index, (candidate, probability) in enumerate(zip(candidates, probabilities, strict=True)):
            if probability >= DETECTOR_THRESHOLD:
                accepted.append((candidate_index, candidate, "detector")); continue
            numeric_text, confidence = _numeric_recognize(scene.raster, candidate.box, numeric)
            role = _role(candidate.box, numeric_text)
            if numeric_text and confidence >= NUMERIC_THRESHOLD and role in {"x_tick", "y_tick"}:
                accepted.append((candidate_index, candidate, "numeric_rescue")); rescue_records[candidate_index] = (numeric_text, confidence, role)
        matched_truths = set(); predictions = []; scene_fp = scene_dup = 0
        for candidate_index, candidate, acceptance_route in accepted:
            matches = [i for i, truth in enumerate(scene.truths) if box_iou(candidate.box, truth.box) >= TRUTH_MATCH_IOU_MINIMUM]
            if not matches: scene_fp += 1; continue
            best = max(matches, key=lambda i: box_iou(candidate.box, scene.truths[i].box))
            if best in matched_truths: scene_dup += 1; continue
            matched_truths.add(best); truth = scene.truths[best]
            raw, conservative_text, official_text, ambiguity_changes = _official_recognize(scene.raster, candidate.box, official, ambiguity, alphabet)
            if candidate_index in rescue_records: numeric_text, confidence, numeric_role = rescue_records[candidate_index]
            else: numeric_text, confidence = _numeric_recognize(scene.raster, candidate.box, numeric); numeric_role = _role(candidate.box, numeric_text)
            official_role = _role(candidate.box, official_text)
            select_numeric = bool(numeric_text) and confidence >= NUMERIC_THRESHOLD and numeric_role in {"x_tick", "y_tick"}
            prediction, predicted_role, route = ((numeric_text, numeric_role, "numeric_specialist") if select_numeric else (official_text, official_role, "general_recognizer"))
            exact = prediction == truth.truth_text; routes[route] += 1
            totals["exact"] += int(exact); totals["errors"] += _distance(truth.truth_text, prediction); totals["characters"] += len(truth.truth_text)
            totals["role_correct"] += int(predicted_role == truth.role); totals["forbidden_numeric"] += int(select_numeric and truth.role not in {"x_tick", "y_tick"})
            totals["spacing_changes"] += int(conservative_text != raw); totals["changed_nonspace"] += int(conservative_text != raw and " " not in truth.truth_text)
            totals["ambiguity_changes"] += ambiguity_changes; totals["rescues"] += int(acceptance_route == "numeric_rescue")
            family_counts[truth.family][0] += int(exact); family_counts[truth.family][1] += 1
            predictions.append({"truth_text": truth.truth_text, "truth_role": truth.role, "text_family": truth.family,
                                "prediction": prediction, "predicted_role": predicted_role, "route": route,
                                "acceptance_route": acceptance_route, "official_raw_prediction": raw,
                                "official_conservative_spacing_prediction": conservative_text,
                                "official_prediction": official_text, "numeric_prediction": numeric_text,
                                "numeric_confidence": confidence, "ambiguity_changed_count": ambiguity_changes,
                                "proposal_bbox": [candidate.box.left, candidate.box.top, candidate.box.right, candidate.box.bottom],
                                "truth_bbox": [truth.box.left, truth.box.top, truth.box.right, truth.box.bottom], "exact": exact})
        scene_fn = len(scene.truths) - len(matched_truths)
        totals["tp"] += len(matched_truths); totals["fp"] += scene_fp; totals["fn"] += scene_fn; totals["dup"] += scene_dup
        cases.append({"scene_id": scene.scene_id, "source_raster_sha256": sha256(scene.raster.tobytes()).hexdigest(),
                      "truth_region_count": len(scene.truths), "proposal_count": len(candidates), "accepted_region_count": len(accepted),
                      "true_positives": len(matched_truths), "false_positives": scene_fp, "false_negatives": scene_fn,
                      "duplicate_region_count": scene_dup, "prohibited_structure_hits": scene_fp,
                      "exact_detection": scene_fp == scene_fn == scene_dup == 0, "predictions": predictions})
    total = sum(len(scene.truths) for scene in scenes)
    return {"scene_count": len(scenes), "truth_region_count": total, "exact_detection_scene_count": sum(int(c["exact_detection"]) for c in cases),
            "true_positives": totals["tp"], "false_positives": totals["fp"], "false_negatives": totals["fn"],
            "duplicate_region_count": totals["dup"], "prohibited_structure_hits": totals["fp"],
            "recognition_exact_match": totals["exact"] / max(1, total), "character_error_rate": totals["errors"] / max(1, totals["characters"]),
            "role_accuracy": totals["role_correct"] / max(1, total),
            "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
            "word_exact_match": family_counts["word"][0] / max(1, family_counts["word"][1]),
            "ambiguity_exact_match": family_counts["ambiguity"][0] / max(1, family_counts["ambiguity"][1]),
            "numeric_rescue_count": totals["rescues"], "ambiguity_changed_count": totals["ambiguity_changes"],
            "spacing_changed_count": totals["spacing_changes"], "spacing_changed_nonspace_truth_count": totals["changed_nonspace"],
            "forbidden_numeric_route_count": totals["forbidden_numeric"], "route_counts": dict(sorted(routes.items())),
            "marker_creation_evaluated": False, "cases": cases}


__all__ = ["DirectRunner", "DirectTensorEvidence", "evaluate_scenes"]
