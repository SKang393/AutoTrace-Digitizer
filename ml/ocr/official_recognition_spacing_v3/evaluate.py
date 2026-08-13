# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Run the preregistered conservative-spacing selection candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import time
from typing import Any
import zipfile

import numpy as np
from PIL import Image

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from ml.ocr.official_bakeoff.production_evaluate import (
    _cpu_session, decode_ctc, read_character_alphabet, recognition_tensor,
)
from .protocol import (
    CANDIDATE_ID, GATES, INFERENCE_YAML_SHA256, MODEL_SHA256, REVISION, TASK,
    TRIGGER_REPORT_SHA256,
)
from .spacing import restore_conservative_source_spaces


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/official_recognition_spacing_v3")
CONFIG_PATH = ROOT / "training/p1.json"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
SELECTION_SEAL_PATH = ROOT / "SELECTION_SEAL.json"
PUBLIC_SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
PUBLIC_GATE_CONFIG_PATH = ROOT / "gates/sealed-public-p1.json"
MODEL_PATH = Path("ml/ocr/official_bakeoff/runs/conversion/en_PP-OCRv5_mobile_rec.onnx")
INFERENCE_YAML_PATH = Path("ml/ocr/official_bakeoff/runs/extracted/en_PP-OCRv5_mobile_rec_infer/inference.yml")
CONVERSION_REPORT_PATH = Path("ml/ocr/official_bakeoff/runs/conversion/report.json")
TRIGGER_REPORT_PATH = Path("ml/ocr/production_composition_v3/VALIDATION_REPORT.json")
CANONICAL_OUTPUT = ROOT / "artifacts/P1-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "prepare_split.py", ROOT / "spacing.py", ROOT / "evaluate.py",
    Path("ml/ocr/official_bakeoff/production_evaluate.py"), Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?%?$")
ROLE_TEXT = {
    **{item: "participant" for item in ("Smith", "Jordan", "Rivera")},
    **{item: "phase_header" for item in ("Baseline", "Intervention", "Maintenance", "Phase A", "Phase B")},
    **{item: "axis_title" for item in ("Session", "Percent", "Frequency")},
    **{item: "legend_text" for item in ("Target", "Level", "Series", "Data Series")},
    **{item: "annotation" for item in (
        "Treatment", "Followup", "Probe", "Condition", "Rule One", "Probe Set", "Low High",
        "Phase Note", "Follow Up", "O o l I", "l I O o", "I l o O", "o O I l",
        "A B C", "A BC", "AB C", "X Y Z", "X YZ", "XY Z",
    )},
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _nonspace(text: str) -> str:
    return "".join(character for character in text if not character.isspace())


def _distance(left: str, right: str) -> int:
    prior = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(current[-1] + 1, prior[right_index] + 1,
                               prior[right_index - 1] + (left_character != right_character)))
        prior = current
    return prior[-1]


def _role(text: str) -> str:
    return "numeric_text" if NUMERIC.fullmatch(text) else ROLE_TEXT.get(text, "other")


def _load_partition(seal_path: Path, partition: str) -> tuple[dict[str, Any], bytes]:
    seal = _load(REPO_ROOT / seal_path)
    if seal.get("partition") != partition:
        raise RuntimeError("Conservative spacing split partition changed")
    manifest_path = REPO_ROOT / str(seal["private_manifest_path"])
    archive_path = REPO_ROOT / str(seal["fixture_archive_path"])
    if sha256_file(manifest_path) != seal["private_manifest_sha256"] or sha256_file(archive_path) != seal["fixture_archive_sha256"]:
        raise RuntimeError("Conservative spacing split bytes changed")
    manifest = _load(manifest_path)
    archive = archive_path.read_bytes()
    cases = manifest.get("cases")
    if manifest.get("synthetic_only") is not True or manifest.get("private_or_article_images") is not False or not isinstance(cases, list):
        raise RuntimeError("Conservative spacing split scope is invalid")
    with zipfile.ZipFile(BytesIO(archive)) as fixtures:
        if set(fixtures.namelist()) != {str(case["source_path"]) for case in cases}:
            raise RuntimeError("Conservative spacing archive inventory changed")
        for case in cases:
            if _hash_bytes(fixtures.read(str(case["source_path"]))) != case["source_sha256"]:
                raise RuntimeError(f"Conservative spacing fixture changed: {case['case_id']}")
    return manifest, archive


def evaluate_partition(manifest: dict[str, Any], archive: bytes, session: Any, alphabet: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    exact = role_exact = errors = characters = raw_exact = spacing_changes = changed_nonspace = mutations = 0
    input_digest = sha256()
    output_digest = sha256()
    started = time.perf_counter()
    with zipfile.ZipFile(BytesIO(archive)) as fixtures:
        for case in manifest["cases"]:
            source = fixtures.read(case["source_path"])
            with Image.open(BytesIO(source)) as loaded:
                image = loaded.convert("RGB")
            tensor = recognition_tensor(image)
            input_bytes = np.ascontiguousarray(tensor).tobytes(order="C")
            input_digest.update(input_bytes)
            output = np.asarray(session.run([session.get_outputs()[0].name], {
                session.get_inputs()[0].name: tensor,
            })[0], dtype=np.float32)
            output_bytes = np.ascontiguousarray(output).tobytes(order="C")
            output_digest.update(output_bytes)
            raw = decode_ctc(output, alphabet)
            prediction = restore_conservative_source_spaces(image, raw)
            matched = prediction == case["truth_text"]
            predicted_role = _role(prediction)
            role_matched = predicted_role == case["truth_role"]
            changed = prediction != raw
            mutation = _nonspace(prediction) != _nonspace(raw)
            exact += int(matched)
            raw_exact += int(raw == case["truth_text"])
            role_exact += int(role_matched)
            errors += _distance(str(case["truth_text"]), prediction)
            characters += len(str(case["truth_text"]))
            spacing_changes += int(changed)
            changed_nonspace += int(changed and " " not in str(case["truth_text"]))
            mutations += int(mutation)
            family = str(case["text_family"])
            family_counts[family][0] += int(matched)
            family_counts[family][1] += 1
            records.append({
                "case_id": case["case_id"], "source_sha256": case["source_sha256"],
                "truth_text": case["truth_text"], "raw_prediction": raw, "prediction": prediction,
                "truth_role": case["truth_role"], "predicted_role": predicted_role,
                "text_family": family, "spacing_changed": changed, "nonspace_character_mutated": mutation,
                "exact": matched, "role_exact": role_matched,
                "input_tensor_sha256": _hash_bytes(input_bytes), "output_tensor_sha256": _hash_bytes(output_bytes),
            })
    count = len(records)
    word_matches = family_counts["compact_word"][0] + family_counts["spaced_word"][0]
    word_count = family_counts["compact_word"][1] + family_counts["spaced_word"][1]
    metrics: dict[str, Any] = {
        "case_count": count,
        "exact_matches": exact,
        "exact_match": exact / max(1, count),
        "raw_exact_match": raw_exact / max(1, count),
        "character_error_rate": errors / max(1, characters),
        "role_accuracy": role_exact / max(1, count),
        "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
        "compact_word_exact_match": family_counts["compact_word"][0] / max(1, family_counts["compact_word"][1]),
        "spaced_word_exact_match": family_counts["spaced_word"][0] / max(1, family_counts["spaced_word"][1]),
        "word_exact_match": word_matches / max(1, word_count),
        "ambiguity_exact_match": family_counts["ambiguity"][0] / max(1, family_counts["ambiguity"][1]),
        "partial_spacing_exact_match": family_counts["partial_spacing"][0] / max(1, family_counts["partial_spacing"][1]),
        "spacing_changed_count": spacing_changes,
        "spacing_changed_nonspace_truth_count": changed_nonspace,
        "nonspace_character_mutation_count": mutations,
        "inference_calls": count,
        "input_tensor_stream_sha256": input_digest.hexdigest(),
        "output_tensor_stream_sha256": output_digest.hexdigest(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    metrics["passed"] = (
        metrics["exact_match"] >= GATES["exact_match_minimum"]
        and metrics["character_error_rate"] <= GATES["character_error_rate_maximum"]
        and metrics["role_accuracy"] >= GATES["role_accuracy_minimum"]
        and metrics["numeric_exact_match"] >= GATES["numeric_exact_match_minimum"]
        and metrics["word_exact_match"] >= GATES["word_exact_match_minimum"]
        and metrics["ambiguity_exact_match"] >= GATES["ambiguity_exact_match_minimum"]
        and metrics["partial_spacing_exact_match"] >= GATES["partial_spacing_exact_match_minimum"]
        and metrics["spacing_changed_nonspace_truth_count"] == 0
        and metrics["nonspace_character_mutation_count"] == 0
    )
    return {"metrics": metrics, "records": records}


def evaluate_candidate(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError(f"Conservative spacing candidate output exists: {output_dir}")
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    try:
        config = _load(REPO_ROOT / CONFIG_PATH)
        protocol = _load(REPO_ROOT / PROTOCOL_PATH)
        if (
            config.get("task") != TASK or config.get("revision") != REVISION
            or config.get("candidate_id") != CANDIDATE_ID or config.get("model_sha256") != MODEL_SHA256
            or config.get("inference_yaml_sha256") != INFERENCE_YAML_SHA256
        ):
            raise RuntimeError("Conservative spacing candidate identity changed")
        if config.get("optimizer_steps") != 0 or config.get("weights_changed") is not False:
            raise RuntimeError("Conservative spacing candidate cannot train or change weights")
        bundle = source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
        if config.get("expected_runner_source_bundle_sha256") != bundle or protocol.get("runner_source_bundle_sha256") != bundle:
            raise RuntimeError("Conservative spacing source bundle changed")
        for configured, actual, label in (
            (config["protocol_sha256"], sha256_file(REPO_ROOT / PROTOCOL_PATH), "protocol"),
            (config["selection_seal_sha256"], sha256_file(REPO_ROOT / SELECTION_SEAL_PATH), "selection seal"),
            (config["sealed_public_test_seal_sha256"], sha256_file(REPO_ROOT / PUBLIC_SEAL_PATH), "public seal"),
            (config["public_gate_config_sha256"], sha256_file(REPO_ROOT / PUBLIC_GATE_CONFIG_PATH), "public gate config"),
            (config["trigger_report_sha256"], sha256_file(REPO_ROOT / TRIGGER_REPORT_PATH), "trigger report"),
        ):
            if configured != actual:
                raise RuntimeError(f"Conservative spacing {label} changed")
        if sha256_file(REPO_ROOT / TRIGGER_REPORT_PATH) != TRIGGER_REPORT_SHA256:
            raise RuntimeError("Exact production-composition failure evidence changed")
        if sha256_file(REPO_ROOT / MODEL_PATH) != MODEL_SHA256 or sha256_file(REPO_ROOT / INFERENCE_YAML_PATH) != INFERENCE_YAML_SHA256:
            raise RuntimeError("Exact official recognizer payload changed")
        conversion = _load(REPO_ROOT / CONVERSION_REPORT_PATH)
        model = next(item for item in conversion["conversion"]["models"] if item["model_id"] == "en_PP-OCRv5_mobile_rec")
        parity_error = float(model["cpu_parity"]["maximum_absolute_difference"])
        if model["cpu_parity"]["passed"] is not True or parity_error > GATES["conversion_parity_maximum_absolute_error"]:
            raise RuntimeError("Official recognizer conversion parity failed")
        selection_seal = _load(REPO_ROOT / SELECTION_SEAL_PATH)
        manifest, archive = _load_partition(SELECTION_SEAL_PATH, "selection")
        evaluated = evaluate_partition(
            manifest, archive, _cpu_session(REPO_ROOT / MODEL_PATH),
            read_character_alphabet(REPO_ROOT / INFERENCE_YAML_PATH),
        )
        report = {
            "schema": "graphreader.ocr-conservative-spacing-candidate.v1",
            "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if evaluated["metrics"]["passed"] else "failed_selection",
            "optimizer_steps": 0, "weights_changed": False,
            "model_onnx_sha256": MODEL_SHA256, "provider": "CPUExecutionProvider",
            "conversion_parity_maximum_absolute_error": parity_error,
            "trigger_report_sha256": TRIGGER_REPORT_SHA256,
            "fixture_archive_sha256": _hash_bytes(archive),
            "private_manifest_sha256": selection_seal["private_manifest_sha256"],
            "gate_requirements": GATES, **evaluated,
            "public_gate_evaluations": 0, "public_gate_archive_opened": False,
            "marker_creation_evaluated": False, "synthetic_only": True,
            "private_or_article_images": False, "chandler_included": False,
            "production_approval": False, "release_eligible": False,
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=report["status"], report_sha256=sha256_file(report_path))
        return report
    except Exception:
        if not report_path.exists():
            report_path.write_bytes(canonical_json_bytes({
                "schema": "graphreader.ocr-conservative-spacing-candidate.v1",
                "task": TASK, "revision": REVISION, "candidate_id": CANDIDATE_ID,
                "status": "failed_runner", "production_approval": False, "release_eligible": False,
                "public_gate_evaluations": 0, "public_gate_archive_opened": False,
            }))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = evaluate_candidate(REPO_ROOT / args.output)
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
