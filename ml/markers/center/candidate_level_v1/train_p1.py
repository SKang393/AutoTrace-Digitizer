# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use candidate P1 training and validation runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ml.markers.center.candidate_level_v1.dataset import build_selection_scenes, selection_manifest
from ml.markers.center.candidate_level_v1.model import (
    CandidateModelConfig,
    CandidatePatchNet,
    candidate_loss,
)
from ml.markers.center.candidate_level_v1.pipeline import (
    concatenate_examples,
    evaluate_scenes,
    extract_proposals,
    sample_training_examples,
)
from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate


REPO_ROOT = Path(__file__).resolve().parents[4]
TASK = "marker-center"
REVISION = "marker-center-candidate-level-v1"
CANDIDATE_ID = "P1"
CONFIG_PATH = Path("ml/markers/center/candidate_level_v1/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/candidate_level_v1/dataset.py"),
    Path("ml/markers/center/candidate_level_v1/model.py"),
    Path("ml/markers/center/candidate_level_v1/pipeline.py"),
    Path("ml/markers/center/candidate_level_v1/train_p1.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def _configure_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _sha256_canonical(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _torch_runner(model: CandidatePatchNet):
    def run(value: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return model(torch.from_numpy(value)).numpy()

    return run


def train_candidate(output_dir: Path) -> dict[str, object]:
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT,
        task=TASK,
        revision=REVISION,
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    if output_dir.exists():
        raise RuntimeError(f"Candidate output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    seed = int(config["seed"])
    _configure_determinism(seed)
    selection_path = REPO_ROOT / config["selection_manifest_path"]
    if sha256_file(selection_path) != config["selection_manifest_sha256"]:
        raise RuntimeError("Selection manifest checksum does not match preregistration")
    generated_selection = selection_manifest()
    if _sha256_canonical(generated_selection) != config["selection_manifest_sha256"]:
        raise RuntimeError("Selection renderer no longer reproduces the frozen manifest")
    sealed_test_seal = REPO_ROOT / config["sealed_public_test_seal_path"]
    if sha256_file(sealed_test_seal) != config["sealed_public_test_seal_sha256"]:
        raise RuntimeError("Sealed public test seal checksum does not match preregistration")
    sealed_metadata = json.loads(sealed_test_seal.read_text(encoding="utf-8"))
    fixture_archive = REPO_ROOT / sealed_metadata["fixture_archive_path"]
    if sha256_file(fixture_archive) != sealed_metadata["fixture_archive_sha256"]:
        raise RuntimeError("Sealed public fixture archive changed before training")

    generator = torch.Generator().manual_seed(seed + 1)
    training_scenes = build_selection_scenes("train")
    validation_scenes = build_selection_scenes("validation")
    examples = concatenate_examples(
        sample_training_examples(
            scene,
            maximum_negative_per_positive=int(config["maximum_negative_per_positive"]),
            generator=generator,
        )
        for scene in training_scenes
    )
    model = CandidatePatchNet(CandidateModelConfig(seed=seed))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    loss_checkpoints: list[dict[str, object]] = []
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(examples.labels), generator=generator)
        epoch_losses: list[dict[str, float]] = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            raw = model.forward_raw(examples.patches.index_select(0, indices))
            loss, components = candidate_loss(
                raw,
                examples.labels.index_select(0, indices),
                examples.offsets.index_select(0, indices),
                examples.radii.index_select(0, indices),
                positive_weight=float(config["positive_loss_weight"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(components)
        if epoch in {0, epochs // 2, epochs - 1}:
            loss_checkpoints.append(
                {
                    "epoch": epoch + 1,
                    **{
                        key: sum(item[key] for item in epoch_losses) / len(epoch_losses)
                        for key in ("total", "marker", "offset", "radius")
                    },
                }
            )

    model.eval()
    runner = _torch_runner(model)
    threshold_results = []
    for threshold in config["selection_thresholds"]:
        result = evaluate_scenes(validation_scenes, runner, threshold=float(threshold))
        threshold_results.append({"threshold": threshold, "metrics": result})
    selected = max(
        threshold_results,
        key=lambda item: (
            item["metrics"]["exact_scene_count"],
            -item["metrics"]["false_positives"],
            item["metrics"]["f1"],
            item["threshold"],
        ),
    )
    selected_threshold = float(selected["threshold"])
    validation = selected["metrics"]
    gate_passed = (
        validation["exact_scene_count"] == validation["scene_count"]
        and validation["duplicate_count"] == 0
        and validation["prohibited_structure_hits"] == 0
        and validation["false_positives"] == 0
        and validation["false_negatives"] == 0
    )

    checkpoint_path = output_dir / "marker-center-candidate-p1.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.export_contract(),
            "selected_threshold": selected_threshold,
        },
        checkpoint_path,
    )
    onnx_path = output_dir / "marker-center-candidate-p1.onnx"
    parity_patches = extract_proposals(validation_scenes[0].tensor).patches[:128]
    torch.onnx.export(
        model,
        parity_patches,
        onnx_path,
        input_names=[model.contract.input_name],
        output_names=[model.contract.output_name],
        dynamic_axes={
            model.contract.input_name: {0: "candidate_count"},
            model.contract.output_name: {0: "candidate_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    parity_input = parity_patches.numpy().astype(np.float32, copy=False)
    with torch.inference_mode():
        expected = model(parity_patches).numpy()
    actual = session.run(None, {model.contract.input_name: parity_input})[0]
    maximum_absolute_error = float(np.max(np.abs(expected - actual)))
    parity_passed = maximum_absolute_error <= float(config["onnx_parity_tolerance"])

    report: dict[str, object] = {
        "schema": "graphreader.marker-center-candidate-training-report.v1",
        "task": TASK,
        "revision": REVISION,
        "candidate_id": CANDIDATE_ID,
        "status": "selected" if gate_passed and parity_passed else "failed_selection",
        "release_eligible": False,
        "public_gate_evaluations": 0,
        "private_or_article_images": False,
        "synthetic_only": True,
        "chandler_included": False,
        "training_authorization": authorization.binding,
        "training_example_count": len(examples.labels),
        "positive_example_count": int(torch.count_nonzero(examples.labels)),
        "epochs": epochs,
        "seed": seed,
        "loss_checkpoints": loss_checkpoints,
        "threshold_comparisons": threshold_results,
        "selected_threshold": selected_threshold,
        "selection_gate_passed": gate_passed,
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx_path": onnx_path.as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_provider": "CPUExecutionProvider",
        "onnx_parity_maximum_absolute_error": maximum_absolute_error,
        "onnx_parity_tolerance": config["onnx_parity_tolerance"],
        "onnx_parity_passed": parity_passed,
        "tensor_contract": model.export_contract(),
        "selection_manifest_sha256": config["selection_manifest_sha256"],
        "sealed_public_test_seal_sha256": config["sealed_public_test_seal_sha256"],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    report_path = output_dir / "candidate-report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    complete_training_candidate(
        authorization,
        status=str(report["status"]),
        report_sha256=sha256_file(report_path),
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ml/markers/center/artifacts/candidate-level-v1/P1-run"),
    )
    args = parser.parse_args()
    result = train_candidate(REPO_ROOT / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
