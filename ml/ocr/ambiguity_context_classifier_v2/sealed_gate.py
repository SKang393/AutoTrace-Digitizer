# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use direct ONNX gate over unopened line-context glyph fixtures."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import onnxruntime as ort
from PIL import Image

from ml.markers.gate_seal import acquire_gate_seal, canonical_json_bytes, complete_gate_seal, require_committed_sources, sha256_file
from ml.markers.training_budget import CANONICAL_LEDGER_PATH
from .dataset import hash_bytes
from .protocol import GATES, GLYPHS, IMAGE_SIZE, PUBLIC_REVISION, REVISION, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/ambiguity_context_classifier_v2")
SPLIT_CONFIG_PATH = ROOT / "gates/sealed-public-v1.json"
EVALUATOR_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "protocol.py", ROOT / "sealed_gate.py",
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)
GATE_CONFIG = {
    "sealed_accuracy_minimum": GATES["sealed_accuracy_minimum"],
    "sealed_macro_accuracy_minimum": GATES["sealed_macro_accuracy_minimum"],
    "sealed_per_class_accuracy_minimum": GATES["sealed_per_class_accuracy_minimum"],
    "provider": "CPUExecutionProvider",
}


def _load_public(seal: dict[str, object]) -> tuple[np.ndarray, np.ndarray, str]:
    archive_path = REPO_ROOT / str(seal["fixture_archive_path"])
    manifest_path = REPO_ROOT / str(seal["private_manifest_path"])
    if sha256_file(archive_path) != seal["fixture_archive_sha256"] or sha256_file(manifest_path) != seal["private_manifest_sha256"]:
        raise RuntimeError("Line-context ambiguity public split bytes changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    values: list[np.ndarray] = []
    labels: list[int] = []
    with zipfile.ZipFile(BytesIO(archive_path.read_bytes())) as fixtures:
        for sample in manifest["samples"]:
            source = fixtures.read(sample["source_path"])
            if hash_bytes(source) != sample["source_sha256"]:
                raise RuntimeError(f"Line-context public fixture changed: {sample['sample_id']}")
            with Image.open(BytesIO(source)) as image:
                tensor = (1.0 - np.asarray(image.convert("L"), dtype=np.float32) / 255.0)[None, :, :]
            if tensor.shape != (1, IMAGE_SIZE, IMAGE_SIZE):
                raise RuntimeError("Line-context public tensor shape changed")
            values.append(tensor)
            labels.append(int(sample["label"]))
    return np.stack(values).astype(np.float32), np.asarray(labels, dtype=np.int64), str(seal["fixture_archive_sha256"])


def evaluate_public(*, onnx_path: Path, selection_report_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise RuntimeError(f"Line-context ambiguity public output exists: {output_path}")
    require_committed_sources(REPO_ROOT, (*EVALUATOR_SOURCE_PATHS, CANONICAL_LEDGER_PATH))
    ledger = json.loads((REPO_ROOT / CANONICAL_LEDGER_PATH).read_text(encoding="utf-8"))
    entry = next(item for item in ledger["revisions"] if item.get("task") == TASK and item.get("revision") == REVISION)
    report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    report_sha256 = sha256_file(selection_report_path)
    onnx_sha256 = sha256_file(onnx_path)
    if (
        entry.get("status") != "selection_passed_public_preregistered"
        or entry.get("public_gate_authorized") is not True
        or entry.get("public_gate_evaluations") != 0
        or entry.get("public_gate_authorized_onnx_sha256") != onnx_sha256
        or entry.get("public_gate_authorized_selection_report_sha256") != report_sha256
        or report.get("status") != "selected" or report.get("selection_gate_passed") is not True
        or report.get("onnx_parity_passed") is not True or report.get("onnx_sha256") != onnx_sha256
        or report.get("sealed_public_archive_opened") is not False
    ):
        raise RuntimeError("Line-context ambiguity public gate lacks exact authorization")
    split_config = json.loads((REPO_ROOT / SPLIT_CONFIG_PATH).read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / str(split_config["sealed_public_test_seal_path"])
    if sha256_file(seal_path) != split_config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("Line-context ambiguity public seal changed")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    gate = acquire_gate_seal(
        repo_root=REPO_ROOT, task=TASK, revision=PUBLIC_REVISION,
        candidate_hashes={"onnx_sha256": onnx_sha256, "selection_report_sha256": report_sha256},
        dataset_manifest_sha256=str(seal["private_manifest_sha256"]), split_config_path=SPLIT_CONFIG_PATH,
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS, gate_config=GATE_CONFIG,
    )
    started = time.perf_counter()
    values, labels, archive_sha256 = _load_public(seal)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("Line-context ambiguity public gate requires CPUExecutionProvider only")
    input_digest = sha256()
    output_digest = sha256()
    predictions: list[int] = []
    calls = 0
    for start in range(0, len(labels), 128):
        batch = np.ascontiguousarray(values[start:start + 128])
        input_digest.update(batch.tobytes())
        logits = np.asarray(session.run(None, {"glyphs": batch})[0], dtype=np.float32)
        output_digest.update(np.ascontiguousarray(logits).tobytes())
        predictions.extend(np.argmax(logits, axis=1).tolist())
        calls += 1
    predicted = np.asarray(predictions, dtype=np.int64)
    per_class = {glyph: float(np.mean(predicted[labels == index] == index)) for index, glyph in enumerate(GLYPHS)}
    metrics = {
        "sample_count": len(labels), "correct_count": int(np.sum(predicted == labels)),
        "accuracy": float(np.mean(predicted == labels)),
        "macro_accuracy": float(np.mean(list(per_class.values()))), "per_class_accuracy": per_class,
    }
    passed = (
        metrics["accuracy"] >= GATES["sealed_accuracy_minimum"]
        and metrics["macro_accuracy"] >= GATES["sealed_macro_accuracy_minimum"]
        and min(per_class.values()) >= GATES["sealed_per_class_accuracy_minimum"]
    )
    output: dict[str, object] = {
        "schema": "graphreader.ocr-ambiguity-context-public-gate.v1",
        "task": TASK, "revision": PUBLIC_REVISION, "status": "pass" if passed else "fail",
        "evaluation_count": 1, "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
        "onnx_sha256": onnx_sha256,
        "selection_report_path": selection_report_path.relative_to(REPO_ROOT).as_posix(),
        "selection_report_sha256": report_sha256, "fixture_archive_sha256": archive_sha256,
        "private_manifest_sha256": seal["private_manifest_sha256"], "provider": "CPUExecutionProvider",
        "class_order": list(GLYPHS), "metrics": metrics,
        "direct_execution": {"inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(),
                             "output_tensor_stream_sha256": output_digest.hexdigest()},
        "gate_requirements": GATE_CONFIG, "seal_binding": gate.binding, "canonical_seal_key": gate.key,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "production_approval": False, "release_eligible": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(output))
    complete_gate_seal(gate, status=str(output["status"]), report_sha256=sha256_file(output_path))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_public(
        onnx_path=REPO_ROOT / args.onnx,
        selection_report_path=REPO_ROOT / args.selection_report,
        output_path=REPO_ROOT / args.output,
    )
    print(json.dumps({"status": report["status"], "metrics": report["metrics"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
