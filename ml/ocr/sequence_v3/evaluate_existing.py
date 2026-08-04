# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Re-evaluate existing V3 artifacts without training or sealed-test claims."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime
import torch

from ml.ocr.metrics import evaluate_predictions

from .dataset import SlotSample, build_corpus, decode, prepare
from .model import CanonicalSlotRecognizer
from .protocol import OBSERVED_HOLDOUT_COUNT, SEED, TRAIN_COUNT, VALIDATION_COUNT

EXPECTED_CHECKPOINT_SHA256 = "dcd9f00389f9bdaa2513d81fd0adfedc91576873855efb6e2c06f0cb4d82126c"
EXPECTED_ONNX_SHA256 = "dfe7a978789d36f71f02b0bbdae07a17d2d6551efa1a0bd7d369fe22184d7e20"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _materialize(samples: tuple[SlotSample, ...]) -> tuple[torch.Tensor, list[str]]:
    return (
        torch.stack([prepare(sample)[0] for sample in samples]),
        [sample.target_text for sample in samples],
    )


def _evaluate_split(
    model: CanonicalSlotRecognizer,
    session: onnxruntime.InferenceSession,
    inputs: torch.Tensor,
    references: list[str],
) -> dict[str, object]:
    maximum = 0.0
    torch_predictions: list[str] = []
    onnx_predictions: list[str] = []
    started = perf_counter()
    for offset in range(0, len(inputs), 64):
        batch = inputs[offset : offset + 64]
        with torch.inference_mode():
            torch_logits = model(batch)
        onnx_logits = session.run(["output"], {"input": batch.numpy()})[0]
        maximum = max(
            maximum,
            float(np.max(np.abs(torch_logits.detach().numpy() - onnx_logits))),
        )
        torch_predictions.extend(decode(torch_logits))
        onnx_predictions.extend(decode(torch.from_numpy(onnx_logits)))
    elapsed_ms = (perf_counter() - started) * 1000
    return {
        "samples": len(inputs),
        "metrics": asdict(
            evaluate_predictions(zip(references, torch_predictions, strict=True))
        ),
        "onnx_metrics": asdict(
            evaluate_predictions(zip(references, onnx_predictions, strict=True))
        ),
        "decoded_predictions_equal": torch_predictions == onnx_predictions,
        "maximum_absolute_logit_difference": maximum,
        "elapsed_ms": elapsed_ms,
    }


def run(checkpoint: Path, model_path: Path, output: Path) -> dict[str, object]:
    checkpoint_hash = _hash(checkpoint)
    model_hash = _hash(model_path)
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Candidate C checkpoint SHA-256 mismatch")
    if model_hash != EXPECTED_ONNX_SHA256:
        raise ValueError("Candidate C ONNX SHA-256 mismatch")

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    model = CanonicalSlotRecognizer()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    model.eval()
    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    corpus = build_corpus(
        SEED,
        train_count=TRAIN_COUNT,
        validation_count=VALIDATION_COUNT,
        test_count=OBSERVED_HOLDOUT_COUNT,
    )
    validation_inputs, validation_references = _materialize(corpus.validation)
    observed_inputs, observed_references = _materialize(corpus.test)
    validation = _evaluate_split(model, session, validation_inputs, validation_references)
    observed = _evaluate_split(model, session, observed_inputs, observed_references)

    dynamic_batches = {}
    for size in (1, 2, 17):
        batch = validation_inputs[:size]
        with torch.inference_mode():
            torch_logits = model(batch).detach().numpy()
        onnx_logits = session.run(["output"], {"input": batch.numpy()})[0]
        dynamic_batches[str(size)] = {
            "output_shape": list(onnx_logits.shape),
            "maximum_absolute_logit_difference": float(
                np.max(np.abs(torch_logits - onnx_logits))
            ),
        }

    maximum = max(
        validation["maximum_absolute_logit_difference"],
        observed["maximum_absolute_logit_difference"],
        *(item["maximum_absolute_logit_difference"] for item in dynamic_batches.values()),
    )
    report = {
        "status": "failed_historical_research_only",
        "sealed_evidence_valid": False,
        "release_eligible": False,
        "candidate_id": "candidate-c",
        "checkpoint_sha256": checkpoint_hash,
        "onnx_sha256": model_hash,
        "provider": "CPUExecutionProvider",
        "representative_scope": "all 512 validation plus all 512 repeatedly observed holdout inputs",
        "validation": validation,
        "observed_nonsealed_holdout": observed,
        "dynamic_batches": dynamic_batches,
        "representative_maximum_absolute_logit_difference": maximum,
        "parity_gate": 1e-4,
        "parity_gate_passed": maximum <= 1e-4,
        "data_scope": "shared procedural generator only; no Chandler or private data",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = run(arguments.checkpoint, arguments.onnx, arguments.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["parity_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
