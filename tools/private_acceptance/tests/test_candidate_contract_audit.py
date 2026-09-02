# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from pathlib import Path

from tools.private_acceptance.candidate_contract_audit import (
    REPO_ROOT,
    TensorSignature,
    _compatible,
    audit,
    inspect_onnx,
)


def test_signature_comparison_rejects_single_head_and_marker_candidate_contracts() -> None:
    ocr_inputs = (
        TensorSignature("proposal_evidence", "float32", (1, "proposal_count", 31)),
        TensorSignature("proposal_crops", "float32", (1, "proposal_count", 2, 32, 128)),
    )
    ocr_outputs = (TensorSignature("proposal_and_role_logits", "float32", (1, "proposal_count", 10)),)
    marker_inputs = (TensorSignature("candidate_patches", "float32", ("candidate_count", 3, 33, 33)),)
    marker_outputs = (TensorSignature("candidate_predictions", "float32", ("candidate_count", 4)),)
    assert not _compatible("ocr-detection-recognition", ocr_inputs, ocr_outputs)
    assert not _compatible("marker-center", marker_inputs, marker_outputs)
    production_input = (TensorSignature("input", "float32", (1, 3, "H", "W")),)
    assert _compatible("marker-center", production_input, (TensorSignature("output", "float32", (1, 3, "H", "W")),))


def test_current_payload_hashes_and_signatures_are_audited_without_inference() -> None:
    result = audit()
    assert result["production_approval"] is False
    assert result["model_inference_runs"] == 0
    assert result["private_corpus_access"] is False
    assert len(result["payloads"]) == 2
    by_task = {item["task"]: item for item in result["payloads"]}
    ocr = by_task["ocr-detection-recognition"]
    assert ocr["adapter_compatible"] is True
    assert ocr["revision"].endswith("composition-v8")
    assert ocr["adapter_factory_path"] == "src/GraphReader.Ocr/OcrV8ProductionCompositionFactory.cs"
    components = {item["kind"]: item for item in ocr["components"]}
    assert set(components) == {
        "detector",
        "official_recognizer",
        "official_recognizer_inference_yaml",
        "numeric_recognizer",
        "ambiguity_recognizer",
    }
    assert components["detector"]["sha256"] == "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db"
    assert components["official_recognizer"]["sha256"] == "7839f12b644f574eaf677e92a11bd3e337f4b2f910160666073888783fece743"
    marker = by_task["marker-center"]
    assert marker["sha256"] == "924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b"
    assert marker["adapter_compatible"] is False
    assert "candidate patches" in marker["compatibility_reason"]
    assert marker["output_signature"][0]["shape"] == ("candidate_count", 4)


def test_inspect_onnx_is_metadata_only_for_known_payload() -> None:
    path = REPO_ROOT / "ml/markers/center/artifacts/runtime-consistency-v2/P2-run/marker-center-runtime-consistency-p2.onnx"
    inputs, outputs = inspect_onnx(path)
    assert inputs[0].name == "candidate_patches"
    assert outputs[0].name == "candidate_predictions"
