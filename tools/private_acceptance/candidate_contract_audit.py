# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Read-only, aggregate-only audit of selected candidate tensor contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESCORE_PATH = REPO_ROOT / "ml/policy/goal22-phase3-rescore-result.json"
OFF = "not_available"


@dataclass(frozen=True)
class TensorSignature:
    name: str
    element_type: str
    shape: tuple[int | str, ...]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _type_name(value: Any) -> str:
    if isinstance(value, str):
        return {"tensor(float)": "float32", "tensor(uint8)": "uint8", "tensor(int32)": "int32", "tensor(int64)": "int64"}.get(value, value)
    mapping = {1: "float32", 2: "uint8", 6: "int32", 7: "int64"}
    return mapping.get(int(value), f"onnx_type_{value}")


def _signature(value: Any) -> TensorSignature:
    shape: list[int | str] = []
    for dimension in value.shape:
        if isinstance(dimension, int):
            shape.append(dimension)
        elif dimension is None:
            shape.append("?")
        else:
            shape.append(str(dimension))
    return TensorSignature(value.name, _type_name(value.type), tuple(shape))


def inspect_onnx(path: Path) -> tuple[tuple[TensorSignature, ...], tuple[TensorSignature, ...]]:
    """Inspect ONNX metadata only.  This function never calls session.run."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("ONNX_RUNTIME_METADATA_DEPENDENCY_MISSING") from error
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return (
        tuple(_signature(item) for item in session.get_inputs()),
        tuple(_signature(item) for item in session.get_outputs()),
    )


def _selected_entry(result: dict[str, Any], task: str) -> dict[str, Any]:
    selected = result["selected_ocr" if task == "ocr-detection-recognition" else "selected_marker"]
    entries = result["ocr_candidates" if task == "ocr-detection-recognition" else "marker_candidates"]
    for entry in entries:
        if entry.get("revision") == selected["revision"] and entry.get("candidate_id") == selected["candidate_id"]:
            return entry
    raise ValueError(f"SELECTED_PAYLOAD_NOT_RECORDED:{task}")


def _compatible(task: str, inputs: tuple[TensorSignature, ...], outputs: tuple[TensorSignature, ...]) -> bool:
    if task == "ocr-detection-recognition":
        return False
    return (
        len(inputs) == 1 and len(outputs) == 1
        and inputs[0].element_type == "float32"
        and inputs[0].shape == (1, 3, "H", "W")
        and outputs[0].element_type == "float32"
        and outputs[0].shape == (1, 3, "H", "W")
    )


def _proposal_adapter_compatible(
    inputs: tuple[TensorSignature, ...],
    outputs: tuple[TensorSignature, ...],
    source: str,
    model_sha256: str,
) -> bool:
    return (
        len(inputs) == 1
        and len(outputs) == 1
        and inputs[0].element_type == "float32"
        and inputs[0].shape == ("candidate_count", 3, 33, 33)
        and outputs[0].element_type == "float32"
        and outputs[0].shape == ("candidate_count", 4)
        and "ProductionProposalMarkerCenterAdapter" in source
        and "DetectCandidateAsync" in source
        and model_sha256 in source
    )


def audit(rescore_path: Path = RESCORE_PATH) -> dict[str, Any]:
    result = json.loads(rescore_path.read_text(encoding="utf-8"))
    payload_audits: list[dict[str, Any]] = []

    ocr = _selected_entry(result, "ocr-detection-recognition")
    components: list[dict[str, Any]] = []
    for payload in ocr["payloads"]:
        path = REPO_ROOT / payload["path"]
        if path.is_file():
            actual_hash = _sha256(path)
            if actual_hash != payload["sha256"]:
                raise ValueError("PAYLOAD_CHECKSUM_MISMATCH:ocr-detection-recognition")
            if path.suffix.lower() == ".onnx":
                inputs, outputs = inspect_onnx(path)
            else:
                inputs, outputs = (), ()
            availability = "present_checksum_verified"
        else:
            actual_hash = None
            inputs, outputs = (), ()
            availability = "missing"
        components.append({
            "kind": payload["kind"],
            "path": payload["path"],
            "sha256": actual_hash,
            "expected_sha256": payload["sha256"],
            "input_signature": [asdict(item) for item in inputs],
            "output_signature": [asdict(item) for item in outputs],
            "availability": availability,
        })
    factory_path = REPO_ROOT / ocr["adapter_factory_path"]
    if _sha256(factory_path) != ocr["adapter_factory_sha256"]:
        raise ValueError("OCR_ADAPTER_FACTORY_CHECKSUM_MISMATCH")
    factory_source = factory_path.read_text(encoding="utf-8")
    model_hashes = [payload["sha256"] for payload in ocr["payloads"] if payload["path"].endswith(".onnx")]
    if any(value not in factory_source for value in model_hashes):
        raise ValueError("OCR_ADAPTER_FACTORY_PAYLOAD_BINDING_MISMATCH")
    payload_audits.append({
        "task": "ocr-detection-recognition",
        "revision": ocr["revision"],
        "candidate_id": ocr["candidate_id"],
        "components": components,
        "adapter_factory_path": ocr["adapter_factory_path"],
        "adapter_factory_sha256": ocr["adapter_factory_sha256"],
        "adapter_compatible": True,
        "compatibility_reason": "The selected four-model V8 composition is checksum-bound by the existing OcrV8ProductionCompositionFactory.",
        "production_approval": False,
        "next_adapter_work": "Run the exact V8 factory on real-dev, then real-sealed, before manifest and store promotion.",
    })

    marker = _selected_entry(result, "marker-center")
    selected_marker = result["selected_marker"]
    payload = next(item for item in marker["payloads"] if item.get("kind") == "onnx")
    path = REPO_ROOT / payload["path"]
    if path.is_file():
        actual_hash = _sha256(path)
        if actual_hash != payload["sha256"]:
            raise ValueError("PAYLOAD_CHECKSUM_MISMATCH:marker-center")
        inputs, outputs = inspect_onnx(path)
        availability = "present_checksum_verified"
    else:
        actual_hash = None
        inputs, outputs = (), ()
        availability = "missing"
    adapter_path = REPO_ROOT / selected_marker["adapter_path"]
    if _sha256(adapter_path) != selected_marker["adapter_sha256"]:
        raise ValueError("MARKER_ADAPTER_CHECKSUM_MISMATCH")
    adapter_source = adapter_path.read_text(encoding="utf-8")
    adapter_compatible = _proposal_adapter_compatible(inputs, outputs, adapter_source, payload["sha256"])
    payload_audits.append({
        "task": "marker-center",
        "revision": marker["revision"],
        "candidate_id": marker["candidate_id"],
        "path": payload["path"],
        "sha256": actual_hash,
        "expected_sha256": payload["sha256"],
        "input_signature": [asdict(item) for item in inputs],
        "output_signature": [asdict(item) for item in outputs],
        "adapter_path": selected_marker["adapter_path"],
        "adapter_sha256": selected_marker["adapter_sha256"],
        "adapter_compatible": adapter_compatible,
        "compatibility_reason": "The checksum-bound candidate adapter implements the selected [N,3,33,33] proposal-patch input and [N,4] output contract.",
        "production_approval": False,
        "availability": availability,
        "next_adapter_work": "Replace or repair the candidate on synthetic train/dev because corrected real-dev marker evidence fails Tier 1.",
    })
    return {
        "schema": "graphreader.candidate-contract-audit.v1",
        "report_scope": "aggregate_only",
        "payloads": payload_audits,
        "production_approval": False,
        "model_inference_runs": 0,
        "private_corpus_access": False,
        "next_work": "Repair OCR and marker candidates on synthetic train/dev; current corrected evidence fails Tier 1 and real-sealed remains unopened.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate-only candidate contract audit")
    parser.add_argument("--rescore", type=Path, default=RESCORE_PATH)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.rescore.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
