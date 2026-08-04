# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Train, export, and seal the canonical-slot V3 experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np
import onnx
import onnxruntime
import torch
from torch import nn

from ml.ocr.metrics import RecognitionMetrics, evaluate_predictions

from .dataset import (
    ALPHABET,
    BLANK_CLASS_INDEX,
    CLASS_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    SLOT_COUNT,
    TIME_STEPS,
    SlotSample,
    build_corpus,
    decode,
    manifest_sha256,
    prepare,
)
from .model import CanonicalSlotRecognizer
from .protocol import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    OBSERVED_HOLDOUT_COUNT,
    PROTOCOL_ID,
    SEED,
    TRAIN_COUNT,
    VALIDATION_COUNT,
    assert_execution_allowed,
    protocol_configuration,
)


@dataclass(frozen=True)
class Evaluation:
    metrics: RecognitionMetrics
    inference_ms: float


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def _materialize(samples: tuple[SlotSample, ...]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    prepared = [prepare(sample) for sample in samples]
    return (
        torch.stack([item[0] for item in prepared]),
        torch.stack([item[1] for item in prepared]),
        [sample.target_text for sample in samples],
    )


def _evaluate(
    model: CanonicalSlotRecognizer,
    inputs: torch.Tensor,
    references: list[str],
) -> Evaluation:
    started = perf_counter()
    with torch.inference_mode():
        predictions = decode(model(inputs))
    return Evaluation(
        evaluate_predictions(zip(references, predictions, strict=True)),
        (perf_counter() - started) * 1000,
    )


def _evaluate_onnx(path: Path, inputs: torch.Tensor, references: list[str]) -> Evaluation:
    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    started = perf_counter()
    logits = session.run(["output"], {"input": inputs.numpy()})[0]
    return Evaluation(
        evaluate_predictions(zip(references, decode(torch.from_numpy(logits)), strict=True)),
        (perf_counter() - started) * 1000,
    )


def run(
    output: Path,
    candidate_id: str,
) -> dict[str, object]:
    configuration = protocol_configuration()
    assert_execution_allowed(candidate_id, configuration)
    seed = SEED
    epochs = EPOCHS
    learning_rate = LEARNING_RATE
    _configure(seed)
    output.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(
        seed,
        train_count=TRAIN_COUNT,
        validation_count=VALIDATION_COUNT,
        test_count=OBSERVED_HOLDOUT_COUNT,
    )
    train_inputs, train_targets, _ = _materialize(corpus.train)
    validation_inputs, _, validation_references = _materialize(corpus.validation)

    model = CanonicalSlotRecognizer()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    batch_size = BATCH_SIZE
    epoch_losses = []
    training_started = perf_counter()
    for epoch in range(epochs):
        ordering = torch.randperm(
            len(corpus.train),
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        loss_total = 0.0
        for offset in range(0, len(ordering), batch_size):
            indices = ordering[offset : offset + batch_size]
            logits = model(train_inputs[indices])[:, ::2, :]
            targets = train_targets[indices, ::2]
            loss = loss_function(logits.reshape(-1, CLASS_COUNT), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
        epoch_losses.append(loss_total / len(corpus.train))
    training_ms = (perf_counter() - training_started) * 1000

    model.eval()
    validation = _evaluate(model, validation_inputs, validation_references)

    checkpoint = output / "graph-numeric-sequence-v3.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "candidate_id": candidate_id,
            "protocol_id": PROTOCOL_ID,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "alphabet": ALPHABET,
        },
        checkpoint,
    )
    model_path = output / "graph-numeric-sequence-v3.onnx"
    example = torch.zeros((2, 1, INPUT_HEIGHT, INPUT_WIDTH), dtype=torch.float32)
    torch.onnx.export(
        model,
        example,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    loaded = onnx.load(str(model_path))
    onnx.checker.check_model(loaded)
    torch_logits = model(example).detach().numpy()
    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(["output"], {"input": example.numpy()})[0]
    parity = float(np.max(np.abs(torch_logits - onnx_logits)))

    # The sealed split is materialized and evaluated only after training,
    # validation selection, export, and parity measurement are complete.
    test_inputs, _, test_references = _materialize(corpus.test)
    heldout = _evaluate(model, test_inputs, test_references)
    onnx_heldout = _evaluate_onnx(model_path, test_inputs, test_references)

    observed_quality_gates_pass = (
        validation.metrics.exact_match >= 0.90
        and heldout.metrics.exact_match >= 0.90
        and heldout.metrics.character_error_rate <= 0.05
        and parity <= 1e-4
        and asdict(heldout.metrics) == asdict(onnx_heldout.metrics)
    )
    report: dict[str, object] = {
        "status": "failed",
        "release_eligible": False,
        "sealed_evidence_valid": False,
        "observed_quality_gates_pass": observed_quality_gates_pass,
        "candidate_id": candidate_id,
        "protocol": configuration,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "architecture": "canonical-glyph-slot-convolutional-sequence-v3",
        "objective": "independent canonical-slot cross entropy; no CTC loss",
        "alphabet": ALPHABET,
        "input_shape": ["N", 1, INPUT_HEIGHT, INPUT_WIDTH],
        "output_shape": ["N", TIME_STEPS, CLASS_COUNT],
        "canonical_slots": SLOT_COUNT,
        "split_sizes": {
            "train": len(corpus.train),
            "validation": len(corpus.validation),
            "test": len(corpus.test),
        },
        "corpus_manifest_sha256": manifest_sha256(corpus),
        "validation": asdict(validation),
        "observed_reused_holdout": asdict(heldout),
        "onnx_observed_reused_holdout": asdict(onnx_heldout),
        "onnx_parity_maximum_absolute_difference": parity,
        "training_ms": training_ms,
        "epoch_losses": epoch_losses,
        "checkpoint": {
            "path": checkpoint.name,
            "bytes": checkpoint.stat().st_size,
            "sha256": _hash(checkpoint),
        },
        "onnx": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _hash(model_path),
        },
        "providers_available_python": onnxruntime.get_available_providers(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(),
        },
        "data_scope": "project-owned procedural vector glyphs only; no private or external data",
        "weights_license": "Apache-2.0",
        "weights_git_eligible": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-id",
        required=True,
        choices=("candidate-a", "candidate-b", "candidate-c"),
    )
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.candidate_id)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
