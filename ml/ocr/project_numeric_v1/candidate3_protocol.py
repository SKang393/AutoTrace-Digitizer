# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Frozen architecture-only protocol for project numeric OCR Candidate 3."""

from __future__ import annotations

import json
from pathlib import Path

from .candidate2_protocol import protocol_configuration as candidate2_configuration
from .protocol import ProtocolViolation

PROTOCOL_ID = "graph-numeric-project-v1-candidate3-20260805"
CANDIDATE_ID = "candidate-3"
CANDIDATE_INDEX = 3
MAXIMUM_CANDIDATES = 3
ARCHITECTURE = "whole-crop-column-self-attention-semantic-query-v1"
CANONICAL_OUTPUT_PATH = Path(__file__).with_name("runs") / CANDIDATE_ID
FROZEN_PROTOCOL_PATH = Path(__file__).with_name("CANDIDATE_3_PROTOCOL.json")


def protocol_configuration() -> dict[str, object]:
    base = candidate2_configuration()
    base.update(
        {
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "candidate_index": CANDIDATE_INDEX,
            "candidate_state": "frozen-and-eligible-after-commit",
            "architecture": ARCHITECTURE,
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "configuration_state": "consumed-rejected",
                    "output_directory": "candidate-1",
                },
                {
                    "candidate_id": "candidate-2",
                    "configuration_state": "consumed-rejected",
                    "output_directory": "candidate-2",
                },
                {
                    "candidate_id": CANDIDATE_ID,
                    "configuration_state": "frozen-and-eligible-after-commit",
                    "output_directory": CANDIDATE_ID,
                },
            ],
            "one_factor_change": {
                "factor": "recognizer architecture",
                "from": "whole-crop-global-spatial-bottleneck-semantic-slot-v1",
                "to": ARCHITECTURE,
                "selection_evidence": "CANDIDATE_3_VALIDATION_DEFECT.json",
                "sealed_results_used": False,
            },
        }
    )
    return base


def load_frozen_protocol() -> dict[str, object]:
    return json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_frozen_protocol() -> dict[str, object]:
    frozen = load_frozen_protocol()
    configuration = frozen.get("configuration")
    if configuration != protocol_configuration():
        raise ProtocolViolation("Candidate 3 frozen protocol does not match code constants.")
    fingerprints = frozen.get("split_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "train",
        "validation",
        "sealed_test",
    }:
        raise ProtocolViolation("Candidate 3 frozen split fingerprints are incomplete.")
    candidate2 = candidate2_configuration()
    allowed_changes = {
        "protocol_id",
        "candidate_id",
        "candidate_index",
        "candidate_state",
        "candidates",
        "architecture",
        "one_factor_change",
    }
    for key in set(candidate2) | set(configuration):
        if key not in allowed_changes and configuration.get(key) != candidate2.get(key):
            raise ProtocolViolation(f"Candidate 3 changed a frozen non-architecture factor: {key}")
    if configuration["maximum_candidates"] != MAXIMUM_CANDIDATES:
        raise ProtocolViolation("Candidate 3 exceeds the fixed candidate budget.")
    return frozen


def assert_candidate_execution_allowed(candidate_id: str, output: Path) -> None:
    validate_frozen_protocol()
    if candidate_id != CANDIDATE_ID:
        raise ProtocolViolation(f"Candidate 3 runner rejects candidate: {candidate_id}")
    if output.resolve() != CANONICAL_OUTPUT_PATH.resolve():
        raise ProtocolViolation(
            f"Candidate 3 output must use the canonical ignored path: {CANONICAL_OUTPUT_PATH}"
        )
    if output.exists():
        raise ProtocolViolation("Candidate 3 output already exists; reruns are prohibited.")
