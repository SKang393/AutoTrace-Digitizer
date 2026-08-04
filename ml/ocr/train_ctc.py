# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Train, export, and evaluate the fixed graph-numeric CTC candidate."""

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

from .ctc_candidate import (
    ALPHABET,
    BLANK_CLASS_INDEX,
    CLASS_COUNT,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    TIME_STEPS,
    CompactGraphNumericCtc,
    CtcSample,
    build_ctc_corpus,
    corpus_manifest_sha256,
    decode_logits,
    encode_target,
    prepare_input,
)
from .metrics import RecognitionMetrics, evaluate_predictions


@dataclass(frozen=True)
class Evaluation:
    metrics: RecognitionMetrics
    inference_ms: float


def _sha256(path: Path) -> str:
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


def _materialize(samples: tuple[CtcSample, ...]) -> tuple[torch.Tensor, list[tuple[int, ...]], list[str]]:
    inputs = torch.stack([prepare_input(sample) for sample in samples])
    targets = [encode_target(sample.target_text) for sample in samples]
    references = [sample.target_text for sample in samples]
    return inputs, targets, references


def _batch_targets(targets: list[tuple[int, ...]], indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [targets[index] for index in indices.tolist()]
    lengths = torch.tensor([len(target) for target in selected], dtype=torch.long)
    flattened = torch.tensor([value for target in selected for value in target], dtype=torch.long)
    return flattened, lengths


def _evaluate_torch(
    model: CompactGraphNumericCtc,
    inputs: torch.Tensor,
    references: list[str],
) -> Evaluation:
    started = perf_counter()
    with torch.inference_mode():
        predictions = decode_logits(model(inputs))
    elapsed = (perf_counter() - started) * 1000
    return Evaluation(evaluate_predictions(zip(references, predictions, strict=True)), elapsed)


def _evaluate_onnx(
    model_path: Path,
    inputs: torch.Tensor,
    references: list[str],
) -> Evaluation:
    session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    started = perf_counter()
    output = session.run(["output"], {"input": inputs.numpy()})[0]
    elapsed = (perf_counter() - started) * 1000
    predictions = decode_logits(torch.from_numpy(output))
    return Evaluation(evaluate_predictions(zip(references, predictions, strict=True)), elapsed)


def run(output: Path, seed: int, epochs: int, learning_rate: float) -> dict[str, object]:
    _configure(seed)
    output.mkdir(parents=True, exist_ok=True)
    corpus = build_ctc_corpus(seed)
    train_inputs, train_targets, _ = _materialize(corpus.train)
    validation_inputs, _, validation_references = _materialize(corpus.validation)
    test_inputs, _, test_references = _materialize(corpus.test)

    model = CompactGraphNumericCtc()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CTCLoss(blank=BLANK_CLASS_INDEX, zero_infinity=True)
    batch_size = 64
    training_started = perf_counter()
    epoch_losses = []
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(seed + epoch)
        ordering = torch.randperm(len(corpus.train), generator=generator)
        model.train()
        total_loss = 0.0
        for offset in range(0, len(ordering), batch_size):
            indices = ordering[offset : offset + batch_size]
            inputs = train_inputs[indices]
            targets, target_lengths = _batch_targets(train_targets, indices)
            logits = model(inputs)
            input_lengths = torch.full((len(indices),), TIME_STEPS, dtype=torch.long)
            loss = loss_function(
                logits.log_softmax(dim=-1).transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        epoch_losses.append(total_loss / len(corpus.train))
    training_ms = (perf_counter() - training_started) * 1000

    model.eval()
    validation = _evaluate_torch(model, validation_inputs, validation_references)
    heldout = _evaluate_torch(model, test_inputs, test_references)

    checkpoint_path = output / "graph-numeric-ctc.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "alphabet": ALPHABET,
        },
        checkpoint_path,
    )
    model_path = output / "graph-numeric-ctc.onnx"
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
    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)

    torch_probe = model(example).detach().numpy()
    ort_session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ort_probe = ort_session.run(["output"], {"input": example.numpy()})[0]
    parity_maximum_absolute_difference = float(np.max(np.abs(torch_probe - ort_probe)))
    onnx_heldout = _evaluate_onnx(model_path, test_inputs, test_references)

    report: dict[str, object] = {
        "status": "candidate" if (
            validation.metrics.exact_match >= 0.90
            and heldout.metrics.exact_match >= 0.90
            and heldout.metrics.character_error_rate <= 0.05
            and parity_maximum_absolute_difference <= 1e-4
        ) else "failed",
        "release_eligible": False,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "architecture": "compact-graph-numeric-ctc-v1",
        "alphabet": ALPHABET,
        "blank_class_index": BLANK_CLASS_INDEX,
        "input_shape": ["N", 1, INPUT_HEIGHT, INPUT_WIDTH],
        "output_shape": ["N", TIME_STEPS, CLASS_COUNT],
        "split_sizes": {
            "train": len(corpus.train),
            "validation": len(corpus.validation),
            "test": len(corpus.test),
        },
        "corpus_manifest_sha256": corpus_manifest_sha256(corpus),
        "validation": asdict(validation),
        "heldout": asdict(heldout),
        "onnx_heldout": asdict(onnx_heldout),
        "training_ms": training_ms,
        "epoch_losses": epoch_losses,
        "onnx_parity_maximum_absolute_difference": parity_maximum_absolute_difference,
        "checkpoint": {
            "path": checkpoint_path.name,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        },
        "onnx": {
            "path": model_path.name,
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
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
        "data_scope": "project-generated procedural graph labels only; no private or external data",
        "weights_git_eligible": False,
    }
    report_path = output / "candidate-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    arguments = parser.parse_args()
    report = run(arguments.output, arguments.seed, arguments.epochs, arguments.learning_rate)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "candidate" else 1


if __name__ == "__main__":
    raise SystemExit(main())
