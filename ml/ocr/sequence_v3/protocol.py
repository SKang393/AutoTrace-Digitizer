# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Immutable V3 protocol and exhausted candidate-budget guard."""

from __future__ import annotations

from dataclasses import dataclass

PROTOCOL_ID = "graph-numeric-sequence-v3-20260804"
PROTOCOL_STATE = "exhausted"
SEED = 20260804
EPOCHS = 24
LEARNING_RATE = 0.002
BATCH_SIZE = 64
TRAIN_COUNT = 2048
VALIDATION_COUNT = 512
OBSERVED_HOLDOUT_COUNT = 512


@dataclass(frozen=True)
class CandidateRegistration:
    candidate_id: str
    output_directory: str
    test_observation: str


CANDIDATES = (
    CandidateRegistration("candidate-a", "candidate-a", "first-and-only-sealed-observation"),
    CandidateRegistration(
        "candidate-b",
        "candidate-b-topology-normalization",
        "reused-nonsealed-observation",
    ),
    CandidateRegistration(
        "candidate-c",
        "candidate-c-topology-columns",
        "reused-nonsealed-observation",
    ),
)


class ProtocolViolation(RuntimeError):
    """Raised before any work when V3 execution would violate its protocol."""


def protocol_configuration() -> dict[str, object]:
    return {
        "evidence_policy": "ml/policy/evidence-policy.json",
        "protocol_id": PROTOCOL_ID,
        "protocol_state": PROTOCOL_STATE,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "train_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
        "observed_holdout_count": OBSERVED_HOLDOUT_COUNT,
        "candidate_ids": [candidate.candidate_id for candidate in CANDIDATES],
    }


def validate_exact_configuration(configuration: dict[str, object]) -> None:
    if configuration != protocol_configuration():
        raise ProtocolViolation("V3 configuration does not exactly match the frozen protocol.")


def assert_execution_allowed(candidate_id: str, configuration: dict[str, object]) -> None:
    validate_exact_configuration(configuration)
    if candidate_id not in {candidate.candidate_id for candidate in CANDIDATES}:
        raise ProtocolViolation(f"Unregistered V3 candidate: {candidate_id}")
    if PROTOCOL_STATE == "exhausted":
        raise ProtocolViolation(
            "V3 candidate budget is exhausted; reruns and fourth candidates are prohibited."
        )
