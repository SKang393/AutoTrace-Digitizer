# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Frozen one-factor protocol for project numeric OCR Candidate 2."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import (
    BATCH_SIZE,
    CER_GATE,
    EPOCHS,
    EXACT_MATCH_GATE,
    LEARNING_RATE,
    MARKER_EXCLUSION_GATE,
    ONNX_PARITY_GATE,
    ROLE_ACCURACY_GATE,
    SEED,
    WEIGHT_DECAY,
    ProtocolViolation,
    protocol_configuration as candidate1_configuration,
)

PROTOCOL_ID = "graph-numeric-project-v1-candidate2-20260805"
CANDIDATE_ID = "candidate-2"
CANDIDATE_INDEX = 2
MAXIMUM_CANDIDATES = 3
CANONICAL_OUTPUT_PATH = Path(__file__).with_name("runs") / CANDIDATE_ID
FROZEN_PROTOCOL_PATH = Path(__file__).with_name("CANDIDATE_2_PROTOCOL.json")
TRAIN_RENDERER_FAMILY = "polyline-domain-randomized-training-v2"
TRAIN_DEGRADATION_FAMILY = "scale-pixelate-thickness-dropout-scanline-v2"
SEALED_RENDERER_FAMILY = "independent-outline-stencil-sealed-v2"
SEALED_DEGRADATION_FAMILY = "contrast-gap-and-offset-sealed-v2"


def protocol_configuration() -> dict[str, object]:
    base = candidate1_configuration()
    splits = [dict(item) for item in base["splits"]]  # type: ignore[arg-type]
    splits[0]["renderer_family"] = TRAIN_RENDERER_FAMILY
    splits[0]["degradation_family"] = TRAIN_DEGRADATION_FAMILY
    splits[2]["renderer_family"] = SEALED_RENDERER_FAMILY
    splits[2]["degradation_family"] = SEALED_DEGRADATION_FAMILY
    base.update(
        {
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "candidate_index": CANDIDATE_INDEX,
            "candidate_state": "frozen-and-eligible-after-commit",
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "configuration_state": "consumed-rejected",
                    "output_directory": "candidate-1",
                },
                {
                    "candidate_id": CANDIDATE_ID,
                    "configuration_state": "frozen-and-eligible-after-commit",
                    "output_directory": CANDIDATE_ID,
                },
                {
                    "candidate_id": "candidate-3",
                    "configuration_state": "reserved-not-registered",
                    "output_directory": "candidate-3",
                },
            ],
            "splits": splits,
            "one_factor_change": {
                "factor": "training renderer and degradation family",
                "from": "independent-polyline-stroke-train-v1 plus train-speckle-and-stroke-variation-v1",
                "to": f"{TRAIN_RENDERER_FAMILY} plus {TRAIN_DEGRADATION_FAMILY}",
                "selection_evidence": "CANDIDATE_2_VALIDATION_DEFECT.json",
                "sealed_results_used": False,
            },
            "sealed_policy": (
                "score and decode only after validation gates pass; raster construction "
                "before training is limited to fingerprint sealing"
            ),
        }
    )
    return base


def load_frozen_protocol() -> dict[str, object]:
    return json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_frozen_protocol() -> dict[str, object]:
    frozen = load_frozen_protocol()
    if frozen.get("configuration") != protocol_configuration():
        raise ProtocolViolation("Candidate 2 frozen protocol does not match code constants.")
    fingerprints = frozen.get("split_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        "train",
        "validation",
        "sealed_test",
    }:
        raise ProtocolViolation("Candidate 2 frozen split fingerprints are incomplete.")
    for value in fingerprints.values():
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolViolation("Candidate 2 split fingerprint is malformed.")
    configuration = frozen["configuration"]
    if (
        configuration["epochs"] != EPOCHS
        or configuration["learning_rate"] != LEARNING_RATE
        or configuration["weight_decay"] != WEIGHT_DECAY
        or configuration["batch_size"] != BATCH_SIZE
        or configuration["maximum_candidates"] != MAXIMUM_CANDIDATES
        or configuration["seed"] != SEED
        or configuration["gates"]["validation_exact_match_minimum"] != EXACT_MATCH_GATE
        or configuration["gates"]["sealed_test_cer_maximum"] != CER_GATE
        or configuration["gates"]["validation_role_accuracy_minimum"] != ROLE_ACCURACY_GATE
        or configuration["gates"]["marker_exclusion_minimum"] != MARKER_EXCLUSION_GATE
        or configuration["gates"]["onnx_parity_maximum_absolute_difference"]
        != ONNX_PARITY_GATE
    ):
        raise ProtocolViolation("Candidate 2 changed a frozen non-renderer factor.")
    return frozen


def assert_candidate_execution_allowed(candidate_id: str, output: Path) -> None:
    validate_frozen_protocol()
    if candidate_id != CANDIDATE_ID:
        raise ProtocolViolation(f"Candidate 2 runner rejects candidate: {candidate_id}")
    if output.resolve() != CANONICAL_OUTPUT_PATH.resolve():
        raise ProtocolViolation(
            f"Candidate 2 output must use the canonical ignored path: {CANONICAL_OUTPUT_PATH}"
        )
    if output.exists():
        raise ProtocolViolation("Candidate 2 output already exists; reruns are prohibited.")
