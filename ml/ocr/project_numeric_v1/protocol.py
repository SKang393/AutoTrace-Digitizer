# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Immutable candidate-1 protocol and fixed public gates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

PROTOCOL_ID = "graph-numeric-project-v1-20260804"
PROTOCOL_STATE = "frozen-preregistration"
CANDIDATE_ID = "candidate-1"
MAXIMUM_CANDIDATES = 3
SEED = 20260804
EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 64
TRAIN_POSITIVE_COUNT = 4096
TRAIN_NEGATIVE_COUNT = 512
VALIDATION_POSITIVE_COUNT = 512
VALIDATION_NEGATIVE_COUNT = 128
SEALED_TEST_POSITIVE_COUNT = 512
SEALED_TEST_NEGATIVE_COUNT = 128
INPUT_HEIGHT = 32
INPUT_WIDTH = 128
TIME_STEPS = 32
MAX_TOKENS = 8
ALPHABET = "0123456789.-%"
CLASS_COUNT = len(ALPHABET) + 1
BLANK_CLASS_INDEX = 0
ROLE_NUMERIC_TEXT = 0
ROLE_NONNUMERIC = 1
ROLE_COUNT = 2
EXACT_MATCH_GATE = 0.90
CER_GATE = 0.05
ROLE_ACCURACY_GATE = 0.90
MARKER_EXCLUSION_GATE = 1.0
ONNX_PARITY_GATE = 1e-4
SLOT_TIME_INDICES = (1, 5, 9, 13, 17, 21, 25, 29)
FROZEN_PROTOCOL_PATH = Path(__file__).with_name("FROZEN_PROTOCOL.json")
CANONICAL_OUTPUT_PATH = Path(__file__).with_name("runs") / CANDIDATE_ID


@dataclass(frozen=True)
class SplitRegistration:
    split: str
    positive_count: int
    negative_count: int
    renderer_family: str
    degradation_family: str
    seed_offset: int


@dataclass(frozen=True)
class CandidateRegistration:
    candidate_id: str
    output_directory: str
    configuration_state: str
    permitted_change: str


SPLITS = (
    SplitRegistration(
        "train",
        TRAIN_POSITIVE_COUNT,
        TRAIN_NEGATIVE_COUNT,
        "independent-polyline-stroke-train-v1",
        "train-speckle-and-stroke-variation-v1",
        11_000,
    ),
    SplitRegistration(
        "validation",
        VALIDATION_POSITIVE_COUNT,
        VALIDATION_NEGATIVE_COUNT,
        "independent-bitmap-validation-v1",
        "validation-fade-and-scanline-v1",
        22_000,
    ),
    SplitRegistration(
        "sealed_test",
        SEALED_TEST_POSITIVE_COUNT,
        SEALED_TEST_NEGATIVE_COUNT,
        "independent-seven-segment-sealed-v1",
        "sealed-contrast-and-dropout-v1",
        33_000,
    ),
)

CANDIDATES = (
    CandidateRegistration(
        "candidate-1",
        "candidate-1",
        "frozen-and-eligible-after-commit",
        "none; exact baseline preregistration",
    ),
    CandidateRegistration(
        "candidate-2",
        "candidate-2",
        "reserved-not-registered",
        "requires a committed validation-only defect report and one frozen change",
    ),
    CandidateRegistration(
        "candidate-3",
        "candidate-3",
        "reserved-not-registered",
        "requires a committed validation-only defect report and one frozen change",
    ),
)


class ProtocolViolation(RuntimeError):
    """Raised before work that would violate the frozen experiment."""


def protocol_configuration() -> dict[str, object]:
    return {
        "protocol_id": PROTOCOL_ID,
        "protocol_state": PROTOCOL_STATE,
        "candidate_id": CANDIDATE_ID,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "input": [1, INPUT_HEIGHT, INPUT_WIDTH],
        "output": [TIME_STEPS, CLASS_COUNT],
        "alphabet": ALPHABET,
        "maximum_tokens": MAX_TOKENS,
        "slot_time_indices": list(SLOT_TIME_INDICES),
        "architecture": "whole-crop-global-spatial-bottleneck-semantic-slot-v1",
        "objective": "fixed semantic-slot cross entropy plus numeric-role cross entropy",
        "maximum_candidates": MAXIMUM_CANDIDATES,
        "candidates": [asdict(candidate) for candidate in CANDIDATES],
        "splits": [asdict(split) for split in SPLITS],
        "gates": {
            "validation_exact_match_minimum": EXACT_MATCH_GATE,
            "sealed_test_exact_match_minimum": EXACT_MATCH_GATE,
            "sealed_test_cer_maximum": CER_GATE,
            "validation_role_accuracy_minimum": ROLE_ACCURACY_GATE,
            "sealed_test_role_accuracy_minimum": ROLE_ACCURACY_GATE,
            "marker_exclusion_minimum": MARKER_EXCLUSION_GATE,
            "onnx_parity_maximum_absolute_difference": ONNX_PARITY_GATE,
            "cpu_execution_required": True,
        },
        "data_scope": (
            "project-owned procedural graph labels and exclusion shapes only; "
            "no private images, external datasets, fonts, or pretrained weights"
        ),
    }


def load_frozen_protocol() -> dict[str, object]:
    return json.loads(FROZEN_PROTOCOL_PATH.read_text(encoding="utf-8"))


def validate_frozen_protocol() -> dict[str, object]:
    frozen = load_frozen_protocol()
    configuration = frozen.get("configuration")
    if configuration != protocol_configuration():
        raise ProtocolViolation("Tracked frozen protocol does not match code constants.")
    fingerprints = frozen.get("split_fingerprints")
    if not isinstance(fingerprints, dict) or set(fingerprints) != {
        split.split for split in SPLITS
    }:
        raise ProtocolViolation("Tracked frozen split fingerprints are incomplete.")
    for value in fingerprints.values():
        if not isinstance(value, str) or len(value) != 64:
            raise ProtocolViolation("Tracked frozen split fingerprint is malformed.")
    return frozen


def assert_candidate_execution_allowed(candidate_id: str, output: Path) -> None:
    validate_frozen_protocol()
    if candidate_id != CANDIDATE_ID:
        raise ProtocolViolation(
            f"Project-numeric candidate is not frozen for execution: {candidate_id}"
        )
    if output.resolve() != CANONICAL_OUTPUT_PATH.resolve():
        raise ProtocolViolation(
            f"Candidate 1 output must use the canonical ignored path: {CANONICAL_OUTPUT_PATH}"
        )
    if output.exists():
        raise ProtocolViolation(
            "Candidate 1 output already exists; reruns and additional candidates are prohibited."
        )
