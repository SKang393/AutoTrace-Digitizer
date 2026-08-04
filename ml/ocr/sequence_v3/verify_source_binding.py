# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Verify the tracked source and historical artifact identities bound to V3."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from .protocol import protocol_configuration

REQUIRED_SOURCE_ROLES = {
    "evaluator": "ml/ocr/sequence_v3/evaluate_existing.py",
    "generator": "ml/ocr/synthetic.py",
    "preprocessor": "ml/ocr/sequence_v3/dataset.py",
    "model": "ml/ocr/sequence_v3/model.py",
    "plan": "ml/ocr/sequence_v3/EXPERIMENT_PLAN.md",
    "protocol": "ml/ocr/sequence_v3/protocol.py",
    "training_entrypoint": "ml/ocr/sequence_v3/train.py",
    "metrics": "ml/ocr/metrics.py",
    "binding_verifier": "ml/ocr/sequence_v3/verify_source_binding.py",
}
REQUIRED_HISTORICAL_ARTIFACTS = {
    "candidate_a_report": "ml/ocr/sequence_v3/runs/candidate-a/report.json",
    "candidate_b_report": (
        "ml/ocr/sequence_v3/runs/candidate-b-topology-normalization/report.json"
    ),
    "candidate_c_report": (
        "ml/ocr/sequence_v3/runs/candidate-c-topology-columns/report.json"
    ),
    "candidate_c_checkpoint": (
        "ml/ocr/sequence_v3/runs/candidate-c-topology-columns/"
        "graph-numeric-sequence-v3.pt"
    ),
    "candidate_c_onnx": (
        "ml/ocr/sequence_v3/runs/candidate-c-topology-columns/"
        "graph-numeric-sequence-v3.onnx"
    ),
    "representative_parity_report": (
        "ml/ocr/sequence_v3/runs/representative-parity.json"
    ),
}


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_historical_artifacts(binding: dict[str, object], repository: Path) -> None:
    artifacts = binding["historical_artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED_HISTORICAL_ARTIFACTS):
        raise ValueError("Historical artifact binding roles do not match the required role set")
    for role, relative in REQUIRED_HISTORICAL_ARTIFACTS.items():
        record = artifacts[role]
        if not isinstance(record, dict) or record.get("path") != relative:
            raise ValueError(f"Historical artifact path mismatch for {role}")
        artifact = repository / relative
        if not artifact.is_file():
            raise ValueError(f"Historical artifact is missing for {role}")
        if artifact.stat().st_size != record.get("bytes"):
            raise ValueError(f"Historical artifact byte count mismatch for {role}")
        if _hash(artifact) != record.get("sha256"):
            raise ValueError(f"Historical artifact SHA-256 mismatch for {role}")


def verify(binding_path: Path | None = None) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[3]
    path = binding_path or Path(__file__).with_name("SOURCE_BINDING.json")
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding["protocol"] != protocol_configuration():
        raise ValueError("Source binding protocol does not match executable protocol")
    if binding["sealed_evidence_valid"] is not False:
        raise ValueError("V3 source binding must retain invalid sealed evidence")
    if binding["family_implementations_independent"] is not False:
        raise ValueError("V3 source binding must disclose shared generator implementations")

    sources = binding["sources"]
    if set(sources) != set(REQUIRED_SOURCE_ROLES):
        raise ValueError("Source binding roles do not match the required role set")
    for role, relative in REQUIRED_SOURCE_ROLES.items():
        record = sources[role]
        if record["path"] != relative:
            raise ValueError(f"Source binding path mismatch for {role}")
        source = repository / relative
        if source.stat().st_size != record["bytes"]:
            raise ValueError(f"Source byte count mismatch for {role}")
        if _hash(source) != record["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch for {role}")
    verify_historical_artifacts(binding, repository)
    return binding


def main() -> int:
    binding = verify()
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
