# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Execute the single preregistered official recognition-only gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable
import zipfile

import numpy as np
from PIL import Image

from ml.markers.gate_seal import (
    acquire_gate_seal,
    canonical_json_bytes,
    complete_gate_seal,
    sha256_file,
    source_bundle_sha256,
)
from ml.markers.training_budget import (
    CANONICAL_LEDGER_PATH,
    acquire_training_candidate,
    complete_training_candidate,
)
from ml.ocr.official_bakeoff.production_evaluate import (
    _cpu_session,
    decode_ctc,
    read_character_alphabet,
    recognition_tensor,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TASK = "ocr-recognition"
REVISION = "official-ppocrv5-recognition-only-v1"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/ocr/official_recognition_v1/training/p1.json")
PROTOCOL_PATH = Path("ml/ocr/official_recognition_v1/PROTOCOL.json")
SELECTION_SEAL_PATH = Path("ml/ocr/official_recognition_v1/SELECTION_SEAL.json")
PUBLIC_SEAL_PATH = Path("ml/ocr/official_recognition_v1/SEALED_PUBLIC_TEST_SEAL.json")
PUBLIC_GATE_CONFIG_PATH = Path("ml/ocr/official_recognition_v1/gates/sealed-public-p1.json")
CANONICAL_OUTPUT = Path("ml/ocr/official_recognition_v1/artifacts/P1-run")
MODEL_PATH = Path("ml/ocr/official_bakeoff/runs/conversion/en_PP-OCRv5_mobile_rec.onnx")
INFERENCE_YAML_PATH = Path("ml/ocr/official_bakeoff/runs/extracted/en_PP-OCRv5_mobile_rec_infer/inference.yml")
CONVERSION_REPORT_PATH = Path("ml/ocr/official_bakeoff/runs/conversion/report.json")
EXPECTED_MODEL_SHA256 = "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/official_recognition_v1/prepare_split.py"),
    Path("ml/ocr/official_recognition_v1/evaluate.py"),
    Path("ml/ocr/official_bakeoff/production_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)
EVALUATOR_SOURCE_PATHS = RUNNER_SOURCE_PATHS
GATES = {
    "exact_match_minimum": 0.90,
    "character_error_rate_maximum": 0.05,
    "role_accuracy_minimum": 0.90,
    "numeric_exact_match_minimum": 0.90,
    "word_exact_match_minimum": 0.90,
    "ambiguity_exact_match_minimum": 0.90,
    "conversion_parity_maximum_absolute_error": 0.0001,
    "provider": "CPUExecutionProvider",
}
PUBLIC_GATE_CONFIG = {
    **GATES,
    "evaluation_limit": 1,
    "production_approval": False,
    "release_eligible": False,
}
NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?%?$")
ROLE_TEXT = {
    "Chandler": "participant",
    "Smith": "participant",
    "Jordan": "participant",
    "Rivera": "participant",
    "Baseline": "phase_header",
    "Intervention": "phase_header",
    "Maintenance": "phase_header",
    "Generalization": "phase_header",
    "Phase A": "phase_header",
    "Phase B": "phase_header",
    "Session": "axis_title",
    "Percent": "axis_title",
    "Frequency": "axis_title",
    "Follow-up": "annotation",
    "Treatment": "annotation",
    "Probe": "annotation",
    "O o l I": "annotation",
}


class RecognitionGateError(RuntimeError):
    """Raised when a frozen recognition gate contract cannot be honored."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecognitionGateError(message)


def _hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RecognitionGateError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _distance(left: str, right: str) -> int:
    prior = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    prior[right_index] + 1,
                    prior[right_index - 1] + (left_character != right_character),
                )
            )
        prior = current
    return prior[-1]


def _predicted_role(text: str) -> str:
    if NUMERIC.fullmatch(text):
        return "numeric_text"
    return ROLE_TEXT.get(text, "other")


def _conversion_parity() -> dict[str, Any]:
    report = _load_json(REPO_ROOT / CONVERSION_REPORT_PATH)
    conversion = report.get("conversion")
    _require(isinstance(conversion, dict), "Conversion report lacks a conversion object.")
    models = conversion.get("models")
    _require(isinstance(models, list), "Conversion report lacks model evidence.")
    model = next(
        (item for item in models if isinstance(item, dict) and item.get("model_id") == "en_PP-OCRv5_mobile_rec"),
        None,
    )
    _require(isinstance(model, dict), "Conversion report lacks recognizer evidence.")
    onnx = model.get("onnx")
    parity = model.get("cpu_parity")
    _require(isinstance(onnx, dict) and isinstance(parity, dict), "Recognizer conversion evidence is incomplete.")
    _require(onnx.get("sha256") == EXPECTED_MODEL_SHA256, "Recognizer conversion hash changed.")
    _require(parity.get("passed") is True, "Recognizer conversion parity failed.")
    maximum = float(parity.get("maximum_absolute_difference", float("inf")))
    _require(maximum <= GATES["conversion_parity_maximum_absolute_error"], "Recognizer conversion parity exceeds the gate.")
    return {
        "conversion_report_sha256": sha256_file(REPO_ROOT / CONVERSION_REPORT_PATH),
        "case_count": int(parity.get("cases", 0)),
        "maximum_absolute_difference": maximum,
        "maximum_allowed": GATES["conversion_parity_maximum_absolute_error"],
        "passed": True,
    }


def _require_public_authorization(selection_report_sha256: str) -> None:
    ledger = _load_json(REPO_ROOT / CANONICAL_LEDGER_PATH)
    entry = next(
        (
            item
            for item in ledger.get("revisions", [])
            if isinstance(item, dict)
            and item.get("task") == TASK
            and item.get("revision") == REVISION
        ),
        None,
    )
    _require(isinstance(entry, dict), "Recognition budget entry is missing.")
    _require(entry.get("public_gate_authorized") is True, "Recognition public gate is not authorized.")
    _require(entry.get("public_gate_authorized_candidate_id") == CANDIDATE_ID, "Recognition public candidate changed.")
    _require(entry.get("public_gate_authorized_on_selection_pass") is True, "Recognition public gate is not conditional on selection.")
    _require(entry.get("public_gate_evaluations") == 0 and entry.get("public_gate_archive_opened") is False, "Recognition public budget is consumed.")
    _require(entry.get("public_gate_authorized_onnx_sha256") == EXPECTED_MODEL_SHA256, "Recognition public model authorization changed.")
    _require(entry.get("selection_report_runtime_binding") == "exact_report_emitted_by_same_committed_p1_run", "Recognition selection report binding changed.")
    _require(len(selection_report_sha256) == 64, "Recognition selection report hash is invalid.")


def _load_partition(seal_path: Path, expected_partition: str) -> tuple[dict[str, Any], bytes]:
    seal = _load_json(REPO_ROOT / seal_path)
    _require(seal.get("partition") == expected_partition, "Recognition split partition changed.")
    manifest_path = REPO_ROOT / str(seal.get("private_manifest_path"))
    archive_path = REPO_ROOT / str(seal.get("fixture_archive_path"))
    _require(manifest_path.is_file() and archive_path.is_file(), "Ignored recognition split evidence is missing.")
    _require(sha256_file(manifest_path) == seal.get("private_manifest_sha256"), "Recognition private manifest changed.")
    _require(sha256_file(archive_path) == seal.get("fixture_archive_sha256"), "Recognition fixture archive changed.")
    manifest = _load_json(manifest_path)
    archive = archive_path.read_bytes()
    _require(manifest.get("schema") == "graphreader.ocr-official-recognition-split.v1", "Recognition split schema changed.")
    _require(manifest.get("partition") == expected_partition, "Recognition manifest partition changed.")
    _require(manifest.get("synthetic_only") is True, "Recognition split must remain synthetic.")
    _require(manifest.get("private_or_article_images") is False, "Private recognition data is prohibited.")
    _require(manifest.get("chandler_included") is False, "Chandler is prohibited from recognition gating.")
    cases = manifest.get("cases")
    _require(isinstance(cases, list) and len(cases) == seal.get("case_count"), "Recognition case count changed.")
    with zipfile.ZipFile(BytesIO(archive), "r") as fixtures:
        names = set(fixtures.namelist())
        _require(len(names) == len(cases), "Recognition archive member count changed.")
        for case in cases:
            _require(isinstance(case, dict), "Recognition case is invalid.")
            source_path = str(case.get("source_path"))
            _require(source_path in names, f"Recognition fixture is missing: {source_path}")
            _require(_hash_bytes(fixtures.read(source_path)) == case.get("source_sha256"), f"Recognition fixture changed: {source_path}")
    return manifest, archive


def _evaluate_partition(
    manifest: dict[str, Any],
    archive: bytes,
    session: Any,
    alphabet: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    exact = role_exact = character_errors = character_count = 0
    family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    started = time.perf_counter()
    input_stream = sha256()
    output_stream = sha256()
    with zipfile.ZipFile(BytesIO(archive), "r") as fixtures:
        for case in manifest["cases"]:
            source = fixtures.read(case["source_path"])
            with Image.open(BytesIO(source)) as loaded:
                image = loaded.convert("RGB")
            tensor = recognition_tensor(image)
            input_stream.update(np.ascontiguousarray(tensor).tobytes(order="C"))
            inference_started = time.perf_counter()
            output = np.asarray(
                session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0],
                dtype=np.float32,
            )
            duration_ms = (time.perf_counter() - inference_started) * 1000.0
            output_stream.update(np.ascontiguousarray(output).tobytes(order="C"))
            prediction = decode_ctc(output, alphabet)
            predicted_role = _predicted_role(prediction)
            matched = prediction == case["truth_text"]
            role_matched = predicted_role == case["truth_role"]
            exact += int(matched)
            role_exact += int(role_matched)
            character_errors += _distance(str(case["truth_text"]), prediction)
            character_count += len(str(case["truth_text"]))
            family = str(case["text_family"])
            family_counts[family][0] += int(matched)
            family_counts[family][1] += 1
            records.append(
                {
                    "case_id": case["case_id"],
                    "source_sha256": case["source_sha256"],
                    "truth_text": case["truth_text"],
                    "prediction": prediction,
                    "truth_role": case["truth_role"],
                    "predicted_role": predicted_role,
                    "exact": matched,
                    "role_exact": role_matched,
                    "input_tensor": {
                        "sha256": _hash_bytes(np.ascontiguousarray(tensor).tobytes(order="C")),
                        "dtype": "float32",
                        "shape": [int(value) for value in tensor.shape],
                    },
                    "output_tensor": {
                        "sha256": _hash_bytes(np.ascontiguousarray(output).tobytes(order="C")),
                        "dtype": "float32",
                        "shape": [int(value) for value in output.shape],
                    },
                    "duration_ms": round(duration_ms, 6),
                }
            )
    count = len(records)
    ambiguity = [record for record in records if record["truth_text"] == "O o l I"]
    metrics = {
        "case_count": count,
        "exact_match": exact / max(1, count),
        "character_error_rate": character_errors / max(1, character_count),
        "role_accuracy": role_exact / max(1, count),
        "numeric_exact_match": family_counts["numeric"][0] / max(1, family_counts["numeric"][1]),
        "word_exact_match": family_counts["word"][0] / max(1, family_counts["word"][1]),
        "ambiguity_exact_match": sum(int(record["exact"]) for record in ambiguity) / max(1, len(ambiguity)),
        "inference_calls": count,
        "input_tensor_stream_sha256": input_stream.hexdigest(),
        "output_tensor_stream_sha256": output_stream.hexdigest(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    metrics["passed"] = (
        metrics["exact_match"] >= GATES["exact_match_minimum"]
        and metrics["character_error_rate"] <= GATES["character_error_rate_maximum"]
        and metrics["role_accuracy"] >= GATES["role_accuracy_minimum"]
        and metrics["numeric_exact_match"] >= GATES["numeric_exact_match_minimum"]
        and metrics["word_exact_match"] >= GATES["word_exact_match_minimum"]
        and metrics["ambiguity_exact_match"] >= GATES["ambiguity_exact_match_minimum"]
    )
    return {"metrics": metrics, "records": records}


def evaluate_candidate(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RecognitionGateError(f"Recognition candidate output already exists: {output_dir}")
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    final_report_path = output_dir / "report.json"
    phase = "preflight"
    public_archive_opened = False
    public_gate_evaluations = 0
    started = time.perf_counter()
    try:
        config = _load_json(REPO_ROOT / CONFIG_PATH)
        protocol = _load_json(REPO_ROOT / PROTOCOL_PATH)
        gate_config = _load_json(REPO_ROOT / PUBLIC_GATE_CONFIG_PATH)
        _require(config.get("optimizer_steps") == 0 and config.get("weights_changed") is False, "Recognition P1 cannot train or change weights.")
        _require(protocol.get("status") == "p1_preregistered_before_inference", "Recognition protocol is not preregistered.")
        _require(protocol.get("production_approval") is False and protocol.get("release_eligible") is False, "Recognition preregistration cannot approve production.")
        runner_bundle = source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS)
        _require(config.get("expected_runner_source_bundle_sha256") == runner_bundle, "Recognition runner source bundle changed.")
        _require(protocol.get("runner_source_bundle_sha256") == runner_bundle, "Recognition protocol source bundle changed.")
        _require(config.get("protocol_sha256") == sha256_file(REPO_ROOT / PROTOCOL_PATH), "Recognition protocol checksum changed.")
        _require(config.get("selection_seal_sha256") == sha256_file(REPO_ROOT / SELECTION_SEAL_PATH), "Recognition selection seal changed.")
        _require(config.get("sealed_public_test_seal_sha256") == sha256_file(REPO_ROOT / PUBLIC_SEAL_PATH), "Recognition public seal changed.")
        _require(config.get("public_gate_config_sha256") == sha256_file(REPO_ROOT / PUBLIC_GATE_CONFIG_PATH), "Recognition public gate config changed.")
        _require(gate_config.get("evaluation_limit") == 1, "Recognition public evaluation limit changed.")
        _require(gate_config.get("production_approval") is False and gate_config.get("release_eligible") is False, "Recognition public config cannot approve production.")
        for key, value in PUBLIC_GATE_CONFIG.items():
            _require(gate_config.get(key) == value, f"Recognition public gate value changed: {key}")
        _require(sha256_file(REPO_ROOT / MODEL_PATH) == EXPECTED_MODEL_SHA256, "Exact recognizer ONNX is missing or changed.")
        _require(sha256_file(REPO_ROOT / INFERENCE_YAML_PATH) == config.get("inference_yaml_sha256"), "Recognizer alphabet metadata changed.")
        parity = _conversion_parity()
        selection_manifest, selection_archive = _load_partition(SELECTION_SEAL_PATH, "selection")
        alphabet = read_character_alphabet(REPO_ROOT / INFERENCE_YAML_PATH)
        session = _cpu_session(REPO_ROOT / MODEL_PATH)
        phase = "selection"
        selection = _evaluate_partition(selection_manifest, selection_archive, session, alphabet)
        selection_report = {
            "schema": "graphreader.ocr-official-recognition-selection.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "pass" if selection["metrics"]["passed"] else "fail",
            "model_sha256": EXPECTED_MODEL_SHA256,
            "fixture_archive_sha256": _hash_bytes(selection_archive),
            "private_manifest_sha256": _hash_bytes(canonical_json_bytes(selection_manifest)),
            "provider": "CPUExecutionProvider",
            "gate_requirements": GATES,
            "conversion_parity": parity,
            **selection,
            "production_approval": False,
            "release_eligible": False,
        }
        selection_path = output_dir / "selection-report.json"
        selection_path.write_bytes(canonical_json_bytes(selection_report))
        public_report: dict[str, Any] | None = None
        public_result_seal_sha256: str | None = None
        if selection["metrics"]["passed"]:
            phase = "public_gate_authorization"
            _require_public_authorization(sha256_file(selection_path))
            public_seal = _load_json(REPO_ROOT / PUBLIC_SEAL_PATH)
            public_manifest_sha256 = str(public_seal.get("private_manifest_sha256"))
            gate = acquire_gate_seal(
                repo_root=REPO_ROOT,
                task=TASK,
                revision=f"{REVISION}-public",
                candidate_hashes={
                    "onnx_sha256": EXPECTED_MODEL_SHA256,
                    "selection_report_sha256": sha256_file(selection_path),
                },
                dataset_manifest_sha256=public_manifest_sha256,
                split_config_path=PUBLIC_GATE_CONFIG_PATH,
                evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
                gate_config=PUBLIC_GATE_CONFIG,
            )
            phase = "public_gate_load"
            public_archive_opened = True
            public_manifest, public_archive = _load_partition(PUBLIC_SEAL_PATH, "sealed_public")
            _require(
                _hash_bytes(canonical_json_bytes(public_manifest)) == public_manifest_sha256,
                "Recognition public manifest does not match the authorized seal.",
            )
            phase = "public_gate_inference"
            public_gate_evaluations = 1
            public = _evaluate_partition(public_manifest, public_archive, session, alphabet)
            public_report = {
                "schema": "graphreader.ocr-official-recognition-public-gate.v1",
                "task": TASK,
                "revision": f"{REVISION}-public",
                "candidate_id": CANDIDATE_ID,
                "status": "pass" if public["metrics"]["passed"] else "fail",
                "model_sha256": EXPECTED_MODEL_SHA256,
                "fixture_archive_sha256": _hash_bytes(public_archive),
                "private_manifest_sha256": _hash_bytes(canonical_json_bytes(public_manifest)),
                "provider": "CPUExecutionProvider",
                "gate_requirements": PUBLIC_GATE_CONFIG,
                "selection_report_sha256": sha256_file(selection_path),
                "seal_binding": gate.binding,
                "canonical_seal_key": gate.key,
                **public,
                "production_approval": False,
                "release_eligible": False,
            }
            public_path = output_dir / "public-report.json"
            public_path.write_bytes(canonical_json_bytes(public_report))
            public_result_seal = complete_gate_seal(
                gate,
                status=str(public_report["status"]),
                report_sha256=sha256_file(public_path),
            )
            public_result_seal_sha256 = sha256_file(public_result_seal)
        passed = bool(selection["metrics"]["passed"] and public_report and public_report["metrics"]["passed"])
        report = {
            "schema": "graphreader.ocr-official-recognition-result.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "public_gate_passed_unapproved" if passed else "failed_gate",
            "model_path": MODEL_PATH.as_posix(),
            "model_sha256": EXPECTED_MODEL_SHA256,
            "optimizer_steps": 0,
            "weights_changed": False,
            "provider": "CPUExecutionProvider",
            "conversion_parity": parity,
            "selection_report_path": selection_path.relative_to(REPO_ROOT).as_posix(),
            "selection_report_sha256": sha256_file(selection_path),
            "selection_metrics": selection["metrics"],
            "public_gate_evaluations": public_gate_evaluations,
            "public_archive_opened": public_archive_opened,
            "public_report_path": (output_dir / "public-report.json").relative_to(REPO_ROOT).as_posix() if public_report else None,
            "public_report_sha256": sha256_file(output_dir / "public-report.json") if public_report else None,
            "public_metrics": public_report["metrics"] if public_report else None,
            "public_result_seal_sha256": public_result_seal_sha256,
            "fixture_byte_execution_bound": True,
            "detector_pair_rerun": False,
            "prior_exposed_pair_splits_used": False,
            "marker_creation_evaluated": False,
            "synthetic_only": True,
            "private_or_article_images": False,
            "chandler_included": False,
            "production_approval": False,
            "release_eligible": False,
            "rerun_allowed": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        final_report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(final_report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-official-recognition-failure.v1",
            "task": TASK,
            "revision": REVISION,
            "candidate_id": CANDIDATE_ID,
            "status": "failed_runner",
            "phase": phase,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "public_gate_evaluations": public_gate_evaluations,
            "public_archive_opened": public_archive_opened,
            "production_approval": False,
            "release_eligible": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        final_report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(final_report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = evaluate_candidate(REPO_ROOT / arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "public_gate_passed_unapproved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
