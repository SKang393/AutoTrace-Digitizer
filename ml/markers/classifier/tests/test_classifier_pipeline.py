# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.markers.classifier.benchmark import benchmark
from ml.markers.classifier.dataset import (
    ARTIFACT_KINDS,
    FILL_NAMES,
    SCENARIOS,
    SHAPE_NAMES,
    SPLIT_FAMILIES,
    SPLIT_TEMPLATES,
    build_fixed_dataset,
    dataset_manifest,
)
from ml.markers.classifier.export import PackedRuntimeClassifier, RUNTIME_OUTPUT_NAME, RUNTIME_SLICES, export_onnx
from ml.markers.classifier.metrics import binary_metrics, classification_metrics, embedding_retrieval_accuracy, supervised_embedding_loss
from ml.markers.classifier.model import CompactMarkerClassifier, save_checkpoint
from ml.markers.classifier.train import LOCAL_FILL_MACRO_F1_GATE, LOCAL_SHAPE_MACRO_F1_GATE, _stack, _train_one_epoch, configure_determinism


def test_split_families_and_templates_are_disjoint_and_contract_coverage_is_complete() -> None:
    family_sets = [set(SPLIT_FAMILIES[split]) for split in ("train", "validation", "test")]
    template_sets = [set(SPLIT_TEMPLATES[split]) for split in ("train", "validation", "test")]
    assert all(left.isdisjoint(right) for index, left in enumerate(family_sets) for right in family_sets[index + 1 :])
    assert all(left.isdisjoint(right) for index, left in enumerate(template_sets) for right in template_sets[index + 1 :])

    samples = tuple(sample for split in ("train", "validation", "test") for sample in build_fixed_dataset(split))
    marker_samples = [sample for sample in samples if sample.artifact < 0.5]
    assert {SHAPE_NAMES[sample.shape_index] for sample in marker_samples} == set(SHAPE_NAMES)
    assert {FILL_NAMES[sample.fill_index] for sample in marker_samples} == set(FILL_NAMES)
    assert {sample.artifact_kind for sample in samples if sample.artifact >= 0.5} == set(ARTIFACT_KINDS)
    assert set(SCENARIOS).issubset({sample.scenario for sample in marker_samples})
    assert any(sample.scenario == "mixed_series_neighbor" for sample in marker_samples)
    assert any(sample.scenario.startswith("line_contact") for sample in marker_samples)
    for shape in ("star", "asterisk", "cross"):
        assert any(SHAPE_NAMES[sample.shape_index] == shape and sample.scenario == "minority_probe" for sample in marker_samples)


def test_fixed_dataset_is_deterministic_and_manifest_can_remain_selection_only() -> None:
    first = build_fixed_dataset("validation")
    second = build_fixed_dataset("validation")
    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert all(torch.equal(left.tensor, right.tensor) for left, right in zip(first, second, strict=True))
    selection_manifest = dataset_manifest(include_test=False)
    assert selection_manifest["included_splits"] == ["train", "validation"]
    assert all(case["split"] != "test" for case in selection_manifest["cases"])
    assert len({case["tensor_sha256"] for case in selection_manifest["cases"]}) > 100


def test_model_has_separate_heads_and_normalized_compact_embedding() -> None:
    configure_determinism()
    model = CompactMarkerClassifier().eval()
    tensor = torch.stack([sample.tensor for sample in build_fixed_dataset("validation")[:7]])
    with torch.inference_mode():
        shape, fill, artifact, embedding = model(tensor)
    assert shape.shape == (7, len(SHAPE_NAMES))
    assert fill.shape == (7, len(FILL_NAMES))
    assert artifact.shape == (7, 1)
    assert embedding.shape == (7, model.config.embedding_size)
    assert torch.allclose(torch.linalg.vector_norm(embedding, dim=1), torch.ones(7), atol=1e-6)
    assert model.config.embedding_size <= 16
    with pytest.raises(ValueError, match="Expected float tensor"):
        model(torch.zeros((1, 3, 32, 32)))


def test_runtime_adapter_packs_heads_in_the_csharp_decoder_order() -> None:
    configure_determinism()
    model = CompactMarkerClassifier().eval()
    shape_temperature = 1.35
    fill_temperature = 0.7
    runtime = PackedRuntimeClassifier(model, shape_temperature, fill_temperature).eval()
    tensor = torch.stack([sample.tensor for sample in build_fixed_dataset("validation")[:3]])
    with torch.inference_mode():
        separate = model(tensor)
        packed = runtime(tensor)
    assert packed.shape == (3, 25)
    expected_runtime = (
        separate[0] / shape_temperature,
        separate[1] / fill_temperature,
        separate[2],
        separate[3],
    )
    for name, expected in zip(model.contract.output_names, expected_runtime, strict=True):
        start, end = RUNTIME_SLICES[name]
        assert torch.equal(packed[:, start:end], expected)


def test_training_step_updates_the_real_model_and_embedding_loss_separates_identities() -> None:
    configure_determinism()
    model = CompactMarkerClassifier()
    samples = build_fixed_dataset("train")[:128]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    before = model.shape_head.weight.detach().clone()
    losses = _train_one_epoch(model, optimizer, _stack(samples), epoch=0)
    assert losses["total"] > 0.0
    assert not torch.equal(before, model.shape_head.weight.detach())

    embeddings = torch.tensor(((1.0, 0.0), (0.98, 0.02), (0.0, 1.0), (0.02, 0.98)), dtype=torch.float32)
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)
    good = supervised_embedding_loss(embeddings, torch.tensor((0, 0, 1, 1)))
    bad = supervised_embedding_loss(embeddings, torch.tensor((0, 1, 0, 1)))
    assert float(good) < float(bad)


def test_metrics_report_macro_f1_calibration_artifacts_and_embedding_retrieval() -> None:
    probabilities = np.array(((0.9, 0.1), (0.2, 0.8), (0.7, 0.3), (0.1, 0.9)), dtype=np.float32)
    targets = np.array((0, 1, 0, 1), dtype=np.int64)
    metrics = classification_metrics(probabilities, targets, 2)
    assert metrics.macro_f1 == 1.0
    assert 0.0 <= metrics.expected_calibration_error <= 1.0
    assert binary_metrics(np.array((0.1, 0.9)), np.array((0.0, 1.0)))["f1"] == 1.0
    embeddings = np.array(((1.0, 0.0), (0.99, 0.01), (0.0, 1.0), (0.01, 0.99)), dtype=np.float32)
    assert embedding_retrieval_accuracy(embeddings, np.array((0, 0, 1, 1))) == 1.0


def test_onnx_export_has_cpu_parity_without_opening_heldout(tmp_path: Path) -> None:
    configure_determinism()
    model = CompactMarkerClassifier().eval()
    checkpoint = tmp_path / "classifier.pt"
    save_checkpoint(
        checkpoint,
        model,
        dataset_manifest_sha256="0" * 64,
        shape_temperature=1.0,
        fill_temperature=1.0,
        training_revision="test",
    )
    report = export_onnx(checkpoint, tmp_path / "classifier.onnx", tmp_path / "parity.json")
    assert report["status"] == "pass"
    assert report["heldout_test_evaluations"] == 0
    assert report["maximum_absolute_error"] <= report["tolerance"]
    assert report["runtime_tensor_contract"]["output_name"] == RUNTIME_OUTPUT_NAME
    assert report["runtime_tensor_contract"]["output_shape"] == ["N", 25]
    assert report["packed_max_abs_error"] <= report["tolerance"]
    assert all(value["pytorch_runtime_transform_exact"] for value in report["slice_parity"].values())
    assert len(report["onnx_sha256"]) == 64


def test_heldout_benchmark_refuses_a_second_open(tmp_path: Path) -> None:
    seal = tmp_path / "heldout-evaluation.seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already opened"):
        benchmark(tmp_path / "missing.pt", tmp_path / "missing.onnx", tmp_path / "benchmark.json")


def test_local_gates_are_explicit_but_not_maintainer_agreement() -> None:
    assert LOCAL_SHAPE_MACRO_F1_GATE == 0.90
    assert LOCAL_FILL_MACRO_F1_GATE == 0.90
    gate_text = (Path(__file__).parents[1] / "ACCEPTANCE_GATE.md").read_text(encoding="utf-8")
    assert "maintainer-agreed" in gate_text
    assert "without representing them as maintainer-agreed" in gate_text
    assert "refuses a second run" in " ".join(gate_text.split())
