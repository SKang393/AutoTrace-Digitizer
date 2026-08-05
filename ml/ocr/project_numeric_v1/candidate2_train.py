# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Run the single preregistered project numeric OCR Candidate 2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .candidate2_dataset import build_candidate2_split
from .candidate2_protocol import (
    CANDIDATE_ID,
    CANONICAL_OUTPUT_PATH,
    assert_candidate_execution_allowed,
    validate_frozen_protocol,
)
from .dataset import prepare_inputs, split_fingerprint, split_metadata
from .model import GlobalSemanticSlotRecognizer
from .protocol import (
    BATCH_SIZE,
    BLANK_CLASS_INDEX,
    CLASS_COUNT,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    WEIGHT_DECAY,
)
from .train import (
    _evaluate,
    _export_and_measure_parity,
    _quality_passes,
    _sha256,
    _tensor_dataset,
)
from .verify_candidate2 import verify_committed_preregistration


def run(output: Path, candidate_id: str) -> dict[str, object]:
    assert_candidate_execution_allowed(candidate_id, output)
    committed_head = verify_committed_preregistration()
    frozen = validate_frozen_protocol()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)

    train_samples = build_candidate2_split("train")
    validation_samples = build_candidate2_split("validation")
    for split_name, samples in (("train", train_samples), ("validation", validation_samples)):
        if split_fingerprint(samples) != frozen["split_fingerprints"][split_name]:
            raise RuntimeError(f"Frozen Candidate 2 {split_name} fingerprint mismatch.")

    output.mkdir(parents=True, exist_ok=False)
    model = GlobalSemanticSlotRecognizer()
    class_weights = torch.ones(CLASS_COUNT)
    class_weights[BLANK_CLASS_INDEX] = 0.25
    slot_loss = nn.CrossEntropyLoss(weight=class_weights)
    role_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loader = DataLoader(
        _tensor_dataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        num_workers=0,
    )
    epoch_losses: list[float] = []
    training_started = time.perf_counter()
    model.train()
    for _ in range(EPOCHS):
        total = 0.0
        batches = 0
        for inputs, slots, roles in loader:
            optimizer.zero_grad(set_to_none=True)
            slot_logits, role_logits = model.semantic_logits(inputs)
            loss = slot_loss(slot_logits.reshape(-1, CLASS_COUNT), slots.reshape(-1))
            loss = loss + 0.35 * role_loss(role_logits, roles)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        epoch_losses.append(total / batches)
    training_ms = (time.perf_counter() - training_started) * 1000.0

    checkpoint = output / "graph-numeric-project-v1-candidate2.pt"
    torch.save(
        {
            "candidate_id": candidate_id,
            "committed_preregistration": committed_head,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    validation = _evaluate(model, validation_samples)
    validation_passed = _quality_passes(validation, sealed=False)

    sealed: dict[str, object]
    onnx_result: dict[str, object] | None = None
    if validation_passed:
        sealed_samples = build_candidate2_split("sealed_test")
        if split_fingerprint(sealed_samples) != frozen["split_fingerprints"]["sealed_test"]:
            raise RuntimeError("Frozen Candidate 2 sealed-test fingerprint mismatch.")
        sealed = _evaluate(model, sealed_samples)
        quality_passed = _quality_passes(sealed, sealed=True)
        if quality_passed:
            onnx_result = _export_and_measure_parity(
                model, output, validation_samples + sealed_samples
            )
    else:
        sealed = {
            "status": "not_evaluated_validation_gate_failed",
            "metrics_opened": False,
            "predictions_opened": False,
            "fingerprint_sha256": frozen["split_fingerprints"]["sealed_test"],
        }
        quality_passed = False
    public_gates_passed = quality_passed and bool(
        onnx_result and onnx_result["gate_passed"]
    )
    report = {
        "protocol_id": frozen["configuration"]["protocol_id"],
        "candidate_id": candidate_id,
        "status": "candidate_public_gates_only" if public_gates_passed else "failed",
        "approved": False,
        "committed_preregistration": committed_head,
        "architecture": frozen["configuration"]["architecture"],
        "one_factor_change": frozen["configuration"]["one_factor_change"],
        "training_time_ms": training_ms,
        "epoch_losses": epoch_losses,
        "split_metadata": {
            "train": split_metadata(train_samples),
            "validation": split_metadata(validation_samples),
            "sealed_test": (
                split_metadata(sealed_samples) if validation_passed else sealed
            ),
        },
        "validation": validation,
        "validation_gate_passed": validation_passed,
        "sealed_test": sealed,
        "onnx": onnx_result,
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
        },
        "weights_license": "Apache-2.0",
        "weights_git_eligible": False,
        "data_scope": frozen["configuration"]["data_scope"],
        "remaining_mandatory_blockers": [
            "downstream zero-marker-creation validation",
            "private graph validation",
            "DirectML provider evidence",
            "production resolver discovery",
            "packaged installer and portable parity",
            "complete release audit",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT_PATH)
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.candidate_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "candidate_public_gates_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
