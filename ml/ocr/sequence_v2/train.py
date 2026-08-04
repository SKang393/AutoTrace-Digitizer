# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Train, export, and seal the spatial sequence V2 experiment."""

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
    TIME_STEPS,
    SequenceSample,
    build_corpus,
    decode,
    manifest_sha256,
    prepare,
)
from .model import SpatialAlignedSequenceModel


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


def _materialize(samples: tuple[SequenceSample, ...]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    prepared = [prepare(sample) for sample in samples]
    return (
        torch.stack([item[0] for item in prepared]),
        torch.stack([item[1] for item in prepared]),
        [sample.target_text for sample in samples],
    )


def _evaluate(
    model: SpatialAlignedSequenceModel,
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
    seed: int = 20260804,
    epochs: int = 36,
    learning_rate: float = 0.002,
    blank_weight: float = 0.20,
    contrast_standardization: bool = False,
) -> dict[str, object]:
    _configure(seed)
    output.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus(seed)
    train_inputs, train_targets, _ = _materialize(corpus.train)
    validation_inputs, _, validation_references = _materialize(corpus.validation)
    test_inputs, _, test_references = _materialize(corpus.test)

    model = SpatialAlignedSequenceModel(contrast_standardization=contrast_standardization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    class_weights = torch.ones(CLASS_COUNT)
    class_weights[BLANK_CLASS_INDEX] = blank_weight
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    batch_size = 64
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
            logits = model(train_inputs[indices])
            loss = loss_function(logits.reshape(-1, CLASS_COUNT), train_targets[indices].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach()) * len(indices)
        epoch_losses.append(loss_total / len(corpus.train))
    training_ms = (perf_counter() - training_started) * 1000

    model.eval()
    validation = _evaluate(model, validation_inputs, validation_references)
    heldout = _evaluate(model, test_inputs, test_references)

    checkpoint = output / "graph-numeric-sequence-v2.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "blank_weight": blank_weight,
            "contrast_standardization": contrast_standardization,
            "alphabet": ALPHABET,
        },
        checkpoint,
    )
    model_path = output / "graph-numeric-sequence-v2.onnx"
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
    onnx_heldout = _evaluate_onnx(model_path, test_inputs, test_references)

    status = "candidate" if (
        validation.metrics.exact_match >= 0.90
        and heldout.metrics.exact_match >= 0.90
        and heldout.metrics.character_error_rate <= 0.05
        and parity <= 1e-4
    ) else "failed"
    report: dict[str, object] = {
        "status": status,
        "release_eligible": False,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "blank_weight": blank_weight,
        "contrast_standardization": contrast_standardization,
        "architecture": "spatial-alignment-supervised-sequence-v2",
        "objective": "dense glyph-span cross entropy; no CTC loss",
        "alphabet": ALPHABET,
        "input_shape": ["N", 1, INPUT_HEIGHT, INPUT_WIDTH],
        "output_shape": ["N", TIME_STEPS, CLASS_COUNT],
        "split_sizes": {
            "train": len(corpus.train),
            "validation": len(corpus.validation),
            "test": len(corpus.test),
        },
        "corpus_manifest_sha256": manifest_sha256(corpus),
        "validation": asdict(validation),
        "heldout": asdict(heldout),
        "onnx_heldout": asdict(onnx_heldout),
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
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--epochs", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--blank-weight", type=float, default=0.20)
    parser.add_argument("--contrast-standardization", action="store_true")
    arguments = parser.parse_args()
    report = run(
        arguments.output,
        arguments.seed,
        arguments.epochs,
        arguments.learning_rate,
        arguments.blank_weight,
        arguments.contrast_standardization,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
