# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Train candidate 1 only after committed preregistration verification."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.ocr.metrics import evaluate_predictions

from .dataset import (
    NumericSample,
    build_split,
    encode_slots,
    prepare_inputs,
    split_fingerprint,
    split_metadata,
)
from .model import GlobalSemanticSlotRecognizer
from .protocol import (
    BATCH_SIZE,
    BLANK_CLASS_INDEX,
    CANONICAL_OUTPUT_PATH,
    CANDIDATE_ID,
    CER_GATE,
    CLASS_COUNT,
    EPOCHS,
    EXACT_MATCH_GATE,
    LEARNING_RATE,
    MARKER_EXCLUSION_GATE,
    ONNX_PARITY_GATE,
    ROLE_ACCURACY_GATE,
    ROLE_NONNUMERIC,
    ROLE_NUMERIC_TEXT,
    SEED,
    WEIGHT_DECAY,
    assert_candidate_execution_allowed,
    validate_frozen_protocol,
)
from .verify_preregistration import verify_committed_preregistration


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def decode_time_logits(logits: torch.Tensor) -> list[str]:
    from .protocol import ALPHABET

    sequences = logits.argmax(dim=-1).detach().cpu().tolist()
    predictions = []
    for sequence in sequences:
        prior = -1
        output = []
        for class_index in sequence:
            if class_index != BLANK_CLASS_INDEX and class_index != prior:
                output.append(ALPHABET[class_index - 1])
            prior = class_index
        predictions.append("".join(output))
    return predictions


def _tensor_dataset(samples: tuple[NumericSample, ...]) -> TensorDataset:
    inputs = prepare_inputs(samples)
    slots = torch.tensor([encode_slots(sample.target_text) for sample in samples], dtype=torch.long)
    roles = torch.tensor([sample.role for sample in samples], dtype=torch.long)
    return TensorDataset(inputs, slots, roles)


def _evaluate(
    model: GlobalSemanticSlotRecognizer,
    samples: tuple[NumericSample, ...],
) -> dict[str, object]:
    model.eval()
    predictions: list[str] = []
    role_predictions: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(samples), BATCH_SIZE):
            batch = samples[offset : offset + BATCH_SIZE]
            time_logits, role_logits = model(prepare_inputs(batch))
            predictions.extend(decode_time_logits(time_logits))
            role_predictions.extend(role_logits.argmax(dim=1).cpu().tolist())
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    positive_indices = [
        index for index, sample in enumerate(samples) if sample.role == ROLE_NUMERIC_TEXT
    ]
    expected = [samples[index].target_text for index in positive_indices]
    observed = [predictions[index] for index in positive_indices]
    metrics = evaluate_predictions(zip(expected, observed, strict=True))
    role_correct = sum(
        int(prediction == sample.role)
        for prediction, sample in zip(role_predictions, samples)
    )
    negative_indices = [
        index for index, sample in enumerate(samples) if sample.role == ROLE_NONNUMERIC
    ]
    excluded = sum(
        int(predictions[index] == "" and role_predictions[index] == ROLE_NONNUMERIC)
        for index in negative_indices
    )
    by_case: dict[str, list[int]] = defaultdict(list)
    for index in positive_indices:
        by_case[samples[index].case].append(int(predictions[index] == samples[index].target_text))
    return {
        "positive_count": len(positive_indices),
        "negative_count": len(negative_indices),
        "exact_match": metrics.exact_match,
        "character_error_rate": metrics.character_error_rate,
        "role_accuracy": role_correct / len(samples),
        "marker_exclusion_accuracy": excluded / len(negative_indices),
        "marker_exclusion_failures": len(negative_indices) - excluded,
        "marker_creation_evaluated": False,
        "marker_creation_gate": "requires downstream application integration",
        "case_exact_match": {
            case: sum(values) / len(values) for case, values in sorted(by_case.items())
        },
        "examples": [
            {
                "sample_id": samples[index].sample_id,
                "expected": samples[index].target_text,
                "predicted": predictions[index],
            }
            for index in positive_indices
            if predictions[index] != samples[index].target_text
        ][:20],
        "cpu_elapsed_ms": elapsed_ms,
        "cpu_ms_per_sample": elapsed_ms / len(samples),
    }


def _quality_passes(result: dict[str, object], sealed: bool) -> bool:
    exact = float(result["exact_match"])
    cer = float(result["character_error_rate"])
    return (
        exact >= EXACT_MATCH_GATE
        and (not sealed or cer <= CER_GATE)
        and float(result["role_accuracy"]) >= ROLE_ACCURACY_GATE
        and float(result["marker_exclusion_accuracy"]) >= MARKER_EXCLUSION_GATE
    )


def _export_and_measure_parity(
    model: GlobalSemanticSlotRecognizer,
    output: Path,
    representative: tuple[NumericSample, ...],
) -> dict[str, object]:
    import onnx
    import onnxruntime

    model_path = output / "graph-numeric-project-v1.onnx"
    model.eval()
    example = prepare_inputs(representative[:2])
    torch.onnx.export(
        model,
        example,
        model_path,
        input_names=["input"],
        output_names=["logits", "role_logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "role_logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(model_path))
    session = onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    maximum = 0.0
    decoded_equal = 0
    observed = 0
    for offset in range(0, len(representative), BATCH_SIZE):
        batch = prepare_inputs(representative[offset : offset + BATCH_SIZE])
        with torch.inference_mode():
            torch_logits, torch_roles = model(batch)
        ort_logits, ort_roles = session.run(None, {"input": batch.numpy()})
        maximum = max(
            maximum,
            float(np.max(np.abs(torch_logits.numpy() - ort_logits))),
            float(np.max(np.abs(torch_roles.numpy() - ort_roles))),
        )
        decoded_equal += sum(
            left == right
            for left, right in zip(
                decode_time_logits(torch_logits),
                decode_time_logits(torch.from_numpy(ort_logits)),
            )
        )
        observed += batch.shape[0]
    dynamic_shapes = {}
    for size in (1, 2, 17):
        input_array = prepare_inputs(representative[:size]).numpy()
        logits, roles = session.run(None, {"input": input_array})
        dynamic_shapes[str(size)] = [list(logits.shape), list(roles.shape)]
    return {
        "path": model_path.name,
        "bytes": model_path.stat().st_size,
        "sha256": _sha256(model_path),
        "provider": "CPUExecutionProvider",
        "maximum_absolute_difference": maximum,
        "decoded_equal_count": decoded_equal,
        "representative_count": observed,
        "dynamic_shapes": dynamic_shapes,
        "gate_passed": maximum <= ONNX_PARITY_GATE and decoded_equal == observed,
    }


def run(output: Path, candidate_id: str) -> dict[str, object]:
    assert_candidate_execution_allowed(candidate_id, output)
    committed_head = verify_committed_preregistration()
    frozen = validate_frozen_protocol()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)

    train_samples = build_split("train")
    validation_samples = build_split("validation")
    for split_name, samples in (("train", train_samples), ("validation", validation_samples)):
        if split_fingerprint(samples) != frozen["split_fingerprints"][split_name]:
            raise RuntimeError(f"Frozen {split_name} split fingerprint mismatch.")

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
    epoch_losses = []
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

    checkpoint = output / "graph-numeric-project-v1.pt"
    torch.save(
        {
            "candidate_id": candidate_id,
            "committed_preregistration": committed_head,
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    validation = _evaluate(model, validation_samples)

    sealed_samples = build_split("sealed_test")
    if split_fingerprint(sealed_samples) != frozen["split_fingerprints"]["sealed_test"]:
        raise RuntimeError("Frozen sealed-test split fingerprint mismatch.")
    sealed = _evaluate(model, sealed_samples)
    quality_passed = _quality_passes(validation, sealed=False) and _quality_passes(
        sealed, sealed=True
    )
    onnx_result = None
    if quality_passed:
        onnx_result = _export_and_measure_parity(
            model, output, validation_samples + sealed_samples
        )
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
        "training_time_ms": training_ms,
        "epoch_losses": epoch_losses,
        "split_metadata": {
            "train": split_metadata(train_samples),
            "validation": split_metadata(validation_samples),
            "sealed_test": split_metadata(sealed_samples),
        },
        "validation": validation,
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


def _load_recovery_record(output: Path, candidate_id: str) -> dict[str, object]:
    if candidate_id != CANDIDATE_ID or output.resolve() != CANONICAL_OUTPUT_PATH.resolve():
        raise RuntimeError("Evaluation recovery is authorized only for canonical Candidate 1.")
    record_path = Path(__file__).with_name("RECOVERY_EVALUATION.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if set(record) != {
        "schema_version",
        "protocol_id",
        "candidate_id",
        "status",
        "original_training_commit",
        "checkpoint",
        "failure",
        "authorized_action",
        "retraining_permitted",
        "weight_changes_permitted",
        "approval",
    }:
        raise RuntimeError("Evaluation recovery record has an unexpected shape.")
    if (
        record["schema_version"] != 1
        or record["protocol_id"] != validate_frozen_protocol()["configuration"]["protocol_id"]
        or record["candidate_id"] != candidate_id
        or record["status"] != "training_complete_evaluation_incomplete"
        or record["retraining_permitted"] is not False
        or record["weight_changes_permitted"] is not False
        or record["approval"] is not False
    ):
        raise RuntimeError("Evaluation recovery authorization is invalid.")
    checkpoint_record = record["checkpoint"]
    if not isinstance(checkpoint_record, dict) or set(checkpoint_record) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise RuntimeError("Evaluation recovery checkpoint record is invalid.")
    checkpoint = output / str(checkpoint_record["path"])
    if not checkpoint.is_file():
        raise RuntimeError("Evaluation recovery checkpoint is missing.")
    if checkpoint.stat().st_size != int(checkpoint_record["bytes"]) or _sha256(
        checkpoint
    ) != str(checkpoint_record["sha256"]):
        raise RuntimeError("Evaluation recovery checkpoint bytes do not match the incident record.")
    unexpected = sorted(path.name for path in output.iterdir() if path != checkpoint)
    if unexpected:
        raise RuntimeError(
            "Evaluation recovery refuses an ambiguous candidate directory: "
            + ", ".join(unexpected)
        )
    return record


def evaluate_recovery(output: Path, candidate_id: str) -> dict[str, object]:
    recovery_commit = verify_committed_preregistration()
    frozen = validate_frozen_protocol()
    record = _load_recovery_record(output, candidate_id)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(8)
    checkpoint_record = record["checkpoint"]
    assert isinstance(checkpoint_record, dict)
    checkpoint = output / str(checkpoint_record["path"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(payload) != {"candidate_id", "committed_preregistration", "state_dict"}:
        raise RuntimeError("Evaluation recovery checkpoint payload has an unexpected shape.")
    if (
        payload["candidate_id"] != candidate_id
        or payload["committed_preregistration"] != record["original_training_commit"]
        or not isinstance(payload["state_dict"], dict)
    ):
        raise RuntimeError("Evaluation recovery checkpoint identity is invalid.")
    model = GlobalSemanticSlotRecognizer()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    validation_samples = build_split("validation")
    sealed_samples = build_split("sealed_test")
    for split_name, samples in (
        ("validation", validation_samples),
        ("sealed_test", sealed_samples),
    ):
        if split_fingerprint(samples) != frozen["split_fingerprints"][split_name]:
            raise RuntimeError(f"Frozen {split_name} split fingerprint mismatch.")
    validation = _evaluate(model, validation_samples)
    sealed = _evaluate(model, sealed_samples)
    quality_passed = _quality_passes(validation, sealed=False) and _quality_passes(
        sealed, sealed=True
    )
    onnx_result = None
    if quality_passed:
        onnx_result = _export_and_measure_parity(
            model, output, validation_samples + sealed_samples
        )
    public_gates_passed = quality_passed and bool(
        onnx_result and onnx_result["gate_passed"]
    )
    report = {
        "protocol_id": frozen["configuration"]["protocol_id"],
        "candidate_id": candidate_id,
        "status": "candidate_public_gates_only" if public_gates_passed else "failed",
        "approved": False,
        "original_training_commit": record["original_training_commit"],
        "recovery_evaluation_commit": recovery_commit,
        "architecture": frozen["configuration"]["architecture"],
        "training_time_ms": None,
        "epoch_losses": None,
        "training_metrics_note": "unavailable because the original process crashed before its report was written",
        "evaluation_recovery": {
            "status": "completed_without_retraining",
            "incident_record": "RECOVERY_EVALUATION.json",
            "checkpoint_sha256": checkpoint_record["sha256"],
            "optimizer_steps": 0,
            "weights_changed": False,
        },
        "split_metadata": {
            "validation": split_metadata(validation_samples),
            "sealed_test": split_metadata(sealed_samples),
        },
        "validation": validation,
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
            "downstream no-marker-creation integration gate",
            "private graph validation",
            "DirectML provider evidence",
            "production resolver discovery",
            "packaged installer and portable parity",
            "complete release audit",
        ],
    }
    report_path = output / "report.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default=CANDIDATE_ID)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("runs") / CANDIDATE_ID,
    )
    parser.add_argument("--resume-evaluation-only", action="store_true")
    arguments = parser.parse_args()
    report = (
        evaluate_recovery(arguments.output, arguments.candidate_id)
        if arguments.resume_evaluation_only
        else run(arguments.output, arguments.candidate_id)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "candidate_public_gates_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
