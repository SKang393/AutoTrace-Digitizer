# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Opt-in aggregate-only private acceptance for approved model candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

from ml.ocr.component_region_detector_v6.dataset import Box
from ml.ocr.crop_evidence_role_anchor_v24.train_p1 import (
    _calibrated_records,
    _cpu_session,
)
from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from ml.ocr.relational_scene_proposal_role_v21.dataset import SceneSample
from ml.ocr.unanimous_structure_veto_v30.pipeline import extract_relational_evidence
from ml.ocr.unanimous_structure_veto_v30.protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
)
from ml.ocr.unanimous_structure_veto_v30.train_p1 import _candidate_session


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_SET_FILE = "private-set.json"
OCR_POLICY_PATH = REPO_ROOT / "ml/policy/ocr-product-bar-v1.json"
OCR_RESULT_PATH = REPO_ROOT / "ml/policy/ocr-product-bar-v1-result.json"
MARKER_CLASSIFIER_SHA256 = "26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a"
CI_VARIABLES = (
    "CI", "TF_BUILD", "GITHUB_ACTIONS", "GITLAB_CI", "BUILD_BUILDID",
    "JENKINS_URL", "TEAMCITY_VERSION",
)
SHAPE_ORDER = (
    "circle", "square", "triangle_up", "triangle_down", "diamond",
    "star", "asterisk", "cross", "other",
)
FILL_ORDER = ("filled", "open", "unknown")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _outside_repository(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError:
        return resolved
    raise RuntimeError("PRIVATE_ROOT_INSIDE_REPOSITORY")


def _ci_enabled() -> bool:
    disabled = {"", "0", "false", "no", "off"}
    return any(os.environ.get(name, "").strip().lower() not in disabled for name in CI_VARIABLES)


def _validate_private_set(root: Path) -> tuple[dict[str, Any], Path]:
    manifest = _read_json(root / PRIVATE_SET_FILE)
    if manifest.get("schema") != "graphreader.private-acceptance-set.v1":
        raise RuntimeError("PRIVATE_SET_SCHEMA_INVALID")
    if (
        manifest.get("training_use") is not False
        or manifest.get("git_eligible") is not False
        or manifest.get("redistribution_authorized") is not False
        or manifest.get("privacy_status") != "private"
    ):
        raise RuntimeError("PRIVATE_SET_PROVENANCE_INVALID")
    image_path = (root / manifest["source"]["file"]).resolve(strict=True)
    image_path.relative_to(root)
    if _sha256(image_path) != manifest["source"]["sha256"]:
        raise RuntimeError("PRIVATE_SET_SOURCE_CHECKSUM_MISMATCH")
    return manifest, image_path


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _run_ocr(image: np.ndarray, manifest: dict[str, Any]) -> dict[str, Any]:
    product_policy = _read_json(OCR_POLICY_PATH)
    product_result = _read_json(OCR_RESULT_PATH)
    selected = product_result["selected_candidate"]
    candidate = next(
        item for item in product_policy["candidates"]
        if item["revision"] == selected["revision"] and item["candidate_id"] == selected["candidate_id"]
    )
    for relative, expected in (
        (DETECTOR_PATH, DETECTOR_SHA256),
        (RECOGNIZER_PATH, RECOGNIZER_SHA256),
        (RECOGNIZER_YAML_PATH, RECOGNIZER_YAML_SHA256),
        (candidate["onnx_path"], candidate["onnx_sha256"]),
    ):
        if _sha256(REPO_ROOT / relative) != expected:
            raise RuntimeError("OCR_PRIVATE_MODEL_CHECKSUM_MISMATCH")

    plot = manifest["plot"]
    scene = SceneSample(
        "private",
        "sealed_public",
        "private",
        "private",
        image,
        Box(int(plot["left"]), int(plot["top"]), int(plot["right"]), int(plot["bottom"])),
        (),
    )
    detector_session = _cpu_session(REPO_ROOT / DETECTOR_PATH)
    recognizer_session = _cpu_session(REPO_ROOT / RECOGNIZER_PATH)
    candidate_session = _candidate_session(REPO_ROOT / candidate["onnx_path"])
    detector_input = detector_session.get_inputs()[0].name
    recognizer_input = recognizer_session.get_inputs()[0].name
    alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)

    def detector_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(detector_session.run(None, {
            detector_input: np.ascontiguousarray(values),
        })[0], dtype=np.float32)

    def recognizer_runner(values: np.ndarray) -> np.ndarray:
        return np.asarray(recognizer_session.run(None, {
            recognizer_input: np.ascontiguousarray(values),
        })[0], dtype=np.float32)

    started = time.perf_counter()
    values, crops, _, records, relations, slices, _ = extract_relational_evidence(
        (scene,), detector_runner, recognizer_runner, alphabet,
        mode="evaluate", negative_cap_per_scene=10_000, recognition_batch_size=64,
    )
    outputs: list[np.ndarray] = []
    for scene_index, scene_slice in enumerate(slices):
        actual = np.asarray(candidate_session.run(None, {
            "proposal_evidence": np.ascontiguousarray(values[scene_slice][None, ...]),
            "proposal_crops": np.ascontiguousarray(crops[scene_slice][None, ...]),
            "proposal_relations": np.ascontiguousarray(relations[scene_index][None, ...]),
        })[0], dtype=np.float32)
        outputs.append(actual[0])
    output = np.concatenate(outputs)
    calibrated = _calibrated_records(records, output)
    probability = _softmax(output[:, :2])[:, 1]
    threshold = float(next(
        item["selected_threshold"] for item in product_result["candidates"]
        if item["revision"] == selected["revision"]
    ))
    accepted = [
        record for record, score in zip(calibrated, probability, strict=True)
        if float(score) >= threshold
    ]

    expected_pairs = Counter(
        (str(item["text"]), str(item["role"])) for item in manifest["ocr"]["expected"]
    )
    predicted_pairs = Counter((record.predicted_text, record.predicted_role) for record in accepted)
    expected_text = Counter(text for text, _ in expected_pairs.elements())
    predicted_text = Counter(text for text, _ in predicted_pairs.elements())
    pair_matches = sum((expected_pairs & predicted_pairs).values())
    text_matches = sum((expected_text & predicted_text).values())
    expected_count = sum(expected_pairs.values())
    accepted_count = len(accepted)
    missing = expected_count - text_matches
    false_regions = accepted_count - text_matches
    duplicates = sum(max(0, predicted_pairs[pair] - expected_pairs[pair]) for pair in predicted_pairs)
    truth_characters = sum(len(text) * count for text, count in expected_text.items())
    unmatched_expected_characters = sum(
        len(text) * max(0, count - predicted_text[text]) for text, count in expected_text.items()
    )
    unmatched_predicted_characters = sum(
        len(text) * max(0, count - expected_text[text]) for text, count in predicted_text.items()
    )
    scene_exact = (
        text_matches == pair_matches == expected_count
        and false_regions == duplicates == 0
    )
    metrics = {
        "scene_count": 1,
        "exact_scene_count": int(scene_exact),
        "scene_exact_rate": float(scene_exact),
        "truth_region_count": expected_count,
        "accepted_region_count": accepted_count,
        "true_positives": text_matches,
        "false_positives": false_regions,
        "false_negatives": missing,
        "duplicate_region_count": duplicates,
        "prohibited_structure_hits": false_regions,
        "recognition_exact": text_matches / expected_count,
        "character_error_rate": (
            unmatched_expected_characters + unmatched_predicted_characters
        ) / truth_characters,
        "role_accuracy": pair_matches / expected_count,
    }
    bar = product_policy["acceptance_bar"]
    gates = {
        "scene_exact_rate": metrics["scene_exact_rate"] >= bar["scene_exact_rate_minimum"]["value"],
        "character_error_rate": metrics["character_error_rate"] <= bar["character_error_rate_maximum"]["value"],
        "role_accuracy": metrics["role_accuracy"] >= bar["role_accuracy_minimum"]["value"],
        "prohibited_structure_hits": metrics["prohibited_structure_hits"] <= bar["prohibited_structure_hits_maximum"]["value"],
    }
    return {
        "stage": "ocr-detection-recognition",
        "status": "pass" if all(gates.values()) else "fail",
        "provider": "CPUExecutionProvider",
        "candidate_revision": selected["revision"],
        "candidate_id": selected["candidate_id"],
        "candidate_onnx_sha256": selected["onnx_sha256"],
        "selected_threshold": threshold,
        "metrics": metrics,
        "gates": gates,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _marker_patch(image: np.ndarray, marker: dict[str, Any]) -> np.ndarray:
    width = height = 32
    center_x = float(marker["x"])
    center_y = float(marker["y"])
    half_extent = max(4.0, float(marker["radius"]) * 2.25)
    patch = np.empty((height, width), dtype=np.float32)
    for patch_y in range(height):
        source_y = center_y + ((((patch_y + 0.5) / height) * 2) - 1) * half_extent
        for patch_x in range(width):
            source_x = center_x + ((((patch_x + 0.5) / width) * 2) - 1) * half_extent
            x0 = int(np.floor(source_x))
            y0 = int(np.floor(source_y))
            x1 = min(x0 + 1, image.shape[1] - 1)
            y1 = min(y0 + 1, image.shape[0] - 1)
            x_fraction = source_x - x0
            y_fraction = source_y - y0
            top = image[y0, x0] + (image[y0, x1] - image[y0, x0]) * x_fraction
            bottom = image[y1, x0] + (image[y1, x1] - image[y1, x0]) * x_fraction
            brightness = top + (bottom - top) * y_fraction
            patch[patch_y, patch_x] = 1.0 - brightness
    return patch


def _run_marker_classifier(
    image: np.ndarray, manifest: dict[str, Any], model_path: Path,
) -> dict[str, Any]:
    if _sha256(model_path) != MARKER_CLASSIFIER_SHA256:
        raise RuntimeError("MARKER_PRIVATE_MODEL_CHECKSUM_MISMATCH")
    markers = manifest["markers"]["expected"]
    inputs = np.stack([_marker_patch(image, marker) for marker in markers])[:, None, :, :]
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    output = np.asarray(session.run(None, {session.get_inputs()[0].name: inputs})[0], dtype=np.float32)
    if output.shape != (len(markers), 25):
        raise RuntimeError("MARKER_PRIVATE_OUTPUT_CONTRACT_INVALID")
    predicted_shape = output[:, :9].argmax(axis=1)
    predicted_fill = output[:, 9:12].argmax(axis=1)
    expected_shape = np.asarray([SHAPE_ORDER.index(item["shape"]) for item in markers])
    expected_fill = np.asarray([FILL_ORDER.index(item["fill"]) for item in markers])
    open_mask = expected_fill == FILL_ORDER.index("open")
    metrics = {
        "marker_count": len(markers),
        "shape_correct": int((predicted_shape == expected_shape).sum()),
        "shape_accuracy": float(np.mean(predicted_shape == expected_shape)),
        "fill_correct": int((predicted_fill == expected_fill).sum()),
        "fill_accuracy": float(np.mean(predicted_fill == expected_fill)),
        "open_probe_count": int(open_mask.sum()),
        "open_probe_correct": int((predicted_fill[open_mask] == expected_fill[open_mask]).sum()),
        "open_probe_accuracy": float(np.mean(predicted_fill[open_mask] == expected_fill[open_mask])),
        "maximum_artifact_probability": float(output[:, 12].max()),
    }
    bar = manifest["markers"]["acceptance_bar"]
    gates = {
        "shape_accuracy": metrics["shape_accuracy"] >= bar["shape_accuracy_minimum"],
        "fill_accuracy": metrics["fill_accuracy"] >= bar["fill_accuracy_minimum"],
        "open_probe_accuracy": metrics["open_probe_accuracy"] >= bar["open_probe_accuracy_minimum"],
    }
    return {
        "stage": "marker-classifier",
        "status": "pass" if all(gates.values()) else "fail",
        "provider": "CPUExecutionProvider",
        "model_id": "graph-marker-classifier",
        "model_version": "0.1.0",
        "model_sha256": MARKER_CLASSIFIER_SHA256,
        "metrics": metrics,
        "gates": gates,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def run(private_root: Path, evidence_root: Path, marker_model: Path) -> dict[str, Any]:
    if _ci_enabled():
        raise RuntimeError("PRIVATE_ACCEPTANCE_DISABLED_IN_CI")
    private_root = _outside_repository(private_root)
    evidence_root = evidence_root.resolve()
    if evidence_root == REPO_ROOT or REPO_ROOT in evidence_root.parents:
        raise RuntimeError("PRIVATE_EVIDENCE_ROOT_INSIDE_REPOSITORY")
    evidence_root.mkdir(parents=True, exist_ok=True)
    manifest, image_path = _validate_private_set(private_root)
    source_hash_before = _sha256(image_path)
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    source_hash_after = _sha256(image_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("PRIVATE_SOURCE_MUTATED")

    stages = [
        _run_ocr(np.rint(image * 255).astype(np.uint8), manifest),
        _run_marker_classifier(image, manifest, marker_model.resolve(strict=True)),
    ]
    report = {
        "schema_version": 1,
        "status": "pass" if all(stage["status"] == "pass" for stage in stages) else "fail",
        "input_count": 1,
        "input_sha256": source_hash_before,
        "source_immutable": source_hash_before == source_hash_after,
        "privacy_status": "private",
        "training_use": False,
        "git_eligible": False,
        "redistribution_authorized": False,
        "report_scope": "aggregate_metrics_only",
        "case_level_output": False,
        "prediction_output": False,
        "truth_row_output": False,
        "pixel_output": False,
        "stages": stages,
        "production_approval": False,
        "release_eligible": False,
    }
    report_path = evidence_root / "aggregate-result.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**report, "report_sha256": _sha256(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--explicit-opt-in", action="store_true")
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--marker-classifier-model", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.explicit_opt_in:
        raise RuntimeError("PRIVATE_ACCEPTANCE_EXPLICIT_OPT_IN_REQUIRED")
    result = run(arguments.private_root, arguments.evidence_root, arguments.marker_classifier_model)
    print(json.dumps({
        "status": result["status"],
        "input_count": result["input_count"],
        "report_scope": result["report_scope"],
        "report_sha256": result["report_sha256"],
        "stages": [
            {"stage": item["stage"], "status": item["status"], "metrics": item["metrics"]}
            for item in result["stages"]
        ],
    }, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
