# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Read-only, aggregate-only audit of selected candidate tensor contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
RESCORE_PATH = REPO_ROOT / "ml/policy/goal22-phase3-rescore-result.json"
OFF = "not_available"


@dataclass(frozen=True)
class TensorSignature:
    name: str
    element_type: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class PayloadAudit:
    task: str
    revision: str
    candidate_id: str
    path: str
    sha256: str | None
    expected_sha256: str | None
    input_signature: tuple[TensorSignature, ...]
    output_signature: tuple[TensorSignature, ...]
    adapter_compatible: bool
    compatibility_reason: str
    production_approval: bool
    availability: str
    next_adapter_work: str


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


def _expected(result: dict[str, Any], task: str) -> dict[str, Any]:
    selected = result["selected_ocr" if task == "ocr-detection-recognition" else "selected_marker"]
    entries = result["ocr_candidates" if task == "ocr-detection-recognition" else "marker_candidates"]
    for entry in entries:
        if entry.get("revision") == selected["revision"] and entry.get("candidate_id") == selected["candidate_id"]:
            payloads = entry.get("payloads", [])
            payload = next((item for item in payloads if item.get("kind") == "onnx"), None)
            if payload is None:
                break
            return {
                "task": task,
                "revision": selected["revision"],
                "candidate_id": selected["candidate_id"],
                "path": str(payload["path"]),
                "sha256": str(payload["sha256"]),
            }
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


def audit(rescore_path: Path = RESCORE_PATH) -> dict[str, Any]:
    result = json.loads(rescore_path.read_text(encoding="utf-8"))
    payload_audits: list[PayloadAudit] = []
    for task in ("ocr-detection-recognition", "marker-center"):
        expected = _expected(result, task)
        path = REPO_ROOT / expected["path"]
        if path.is_file():
            actual_hash = _sha256(path)
            if actual_hash != expected["sha256"]:
                raise ValueError(f"PAYLOAD_CHECKSUM_MISMATCH:{task}")
            inputs, outputs = inspect_onnx(path)
            availability = "present_checksum_verified"
        else:
            actual_hash = None
            inputs, outputs = (), ()
            availability = "missing"
        compatible = _compatible(task, inputs, outputs) if inputs else False
        if task.startswith("ocr"):
            reason = (
                "The selected V30 payload is a proposal-evidence/crop/relation head, while the current production OCR adapter composes separate full-image detector and crop-recognizer payloads."
            )
            work = (
                "Implement and review a candidate-specific V17 plus official-recognizer plus V30 proposal adapter, or select a Tier-1 composition already supported by the production OCR adapter."
            )
        else:
            reason = (
                "The selected P2 payload consumes candidate patches and returns four proposal values, while the current marker adapter requires equal bounded full-frame NCHW [1,3,H,W] input and output."
            )
            work = (
                "Implement and review a proposal-patch marker adapter, or provide a compatible full-frame [1,3,H,W] model."
            )
        payload_audits.append(PayloadAudit(
            task=task,
            revision=expected["revision"],
            candidate_id=expected["candidate_id"],
            path=expected["path"],
            sha256=actual_hash,
            expected_sha256=expected["sha256"],
            input_signature=inputs,
            output_signature=outputs,
            adapter_compatible=compatible,
            compatibility_reason=reason,
            production_approval=False,
            availability=availability,
            next_adapter_work=work,
        ))
    return {
        "schema": "graphreader.candidate-contract-audit.v1",
        "report_scope": "aggregate_only",
        "payloads": [asdict(item) for item in payload_audits],
        "production_approval": False,
        "model_inference_runs": 0,
        "private_corpus_access": False,
        "next_work": "Create checksum-bound candidate-specific adapters and manifests; do not weaken the existing production contracts.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate-only candidate contract audit")
    parser.add_argument("--rescore", type=Path, default=RESCORE_PATH)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.rescore.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
