# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from ml.ocr.production_csharp_marker_gate_v1.protocol import (
    DEGRADATION_FAMILIES,
    MODEL_SHA256,
    RENDERER_FAMILIES,
    SCENE_COUNT,
    protocol_configuration,
)
from ml.ocr.production_csharp_marker_gate_v1.dataset import build_scene


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_frozen_fail_closed_and_private_free() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v1/PROTOCOL.json").read_text(encoding="utf-8")
    )
    assert protocol["schema"] == tracked["schema"]
    assert protocol["revision"] == tracked["revision"]
    assert protocol["composition_id"] == tracked["composition_id"]
    assert protocol["state"] == "frozen_before_any_model_execution_on_the_new_split"
    assert protocol["split"]["scene_count"] == tracked["split"]["scene_count"] == SCENE_COUNT
    assert protocol["synthetic_only"] is True
    assert protocol["private_or_article_images"] is False
    assert protocol["chandler_included"] is False
    assert protocol["generalization_label_included"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert protocol["manifest_created"] is False
    assert protocol["model_store_promoted"] is False


def test_models_and_direct_evidence_requirements_are_exact() -> None:
    protocol = protocol_configuration()
    assert len(MODEL_SHA256) == 5
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in MODEL_SHA256.values())
    assert {item["sha256"] for item in protocol["models"].values()} == set(MODEL_SHA256.values())
    assert {item["provider"] for item in protocol["models"].values()} == {"CPUExecutionProvider"}
    gates = protocol["gates"]
    assert gates["direct_fixture_byte_execution_required"] is True
    assert gates["direct_tensor_hash_per_model_call_required"] is True
    assert gates["exact_ocr_region_count_every_fixture"] is True
    assert gates["exact_marker_count_every_fixture"] is True
    assert gates["text_marker_creation_count"] == 0
    assert gates["line_intersection_hits"] == 0


def test_new_composite_families_and_fixture_mask_limit_are_explicit() -> None:
    protocol = protocol_configuration()
    assert len(RENDERER_FAMILIES) == 5 and len(set(RENDERER_FAMILIES)) == 5
    assert len(DEGRADATION_FAMILIES) == 4 and len(set(DEGRADATION_FAMILIES)) == 4
    assert all(value.endswith("-public") or "-public-" in value for value in RENDERER_FAMILIES)
    assert all(value.endswith("-public") for value in DEGRADATION_FAMILIES)
    assert protocol["split"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split"]["secret_seed_generated_once_and_not_serialized"] is True
    assert protocol["composition"]["ocr_text_mask_source"] == (
        "accepted_CSharp_OcrResult_regions_and_masks_only"
    )
    assert protocol["composition"]["artifact_mask_production_approval"] is False
    assert "approved_artifact_mask_provider" in protocol["blocking_gates_after_pass"]


def test_new_scene_has_real_text_marker_and_structure_truth_without_private_labels() -> None:
    scene = build_scene(0x123456789ABCDEF, 0)
    assert scene.raster.shape == (320, 640)
    assert scene.raster.dtype.name == "uint8"
    assert scene.artifact_mask.shape == scene.raster.shape
    assert set(scene.artifact_mask.ravel()).issubset({0, 255})
    assert len(scene.text_truths) == 5
    assert len(scene.marker_centers) >= 8
    assert {item.family for item in scene.text_truths} >= {"numeric", "word"}
    assert {item.kind for item in scene.prohibited} >= {
        "axis", "tick", "divider", "legend", "bracket", "arrowhead"
    }
    combined = " ".join(item.display_text for item in scene.text_truths)
    assert "Chandler" not in combined
    assert "Generalization" not in combined
    assert all(0 <= x < 640 and 0 <= y < 320 for x, y in scene.marker_centers)


def test_tracked_split_seal_is_hash_only_fail_closed_and_has_no_secret() -> None:
    seal_path = REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v1/SEALED_PUBLIC_TEST_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["schema"] == "graphreader.ocr-marker-production-composition-split-seal.v1"
    assert seal["scene_count"] == SCENE_COUNT
    assert seal["text_truth_count"] == SCENE_COUNT * 5
    assert seal["marker_truth_count"] > 0
    assert seal["model_execution_count_at_freeze"] == 0
    assert seal["secret_seed_serialized"] is False
    assert "secret_seed" not in seal
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    assert len(seal["fixture_archive_sha256"]) == 64
    assert len(seal["fixture_manifest_sha256"]) == 64
    assert seal["generation_environment"] == {
        "implementation": "CPython",
        "numpy": "2.3.5",
        "pillow": "12.3.0",
        "platform": "Windows-11-10.0.26200-SP0",
        "python": "3.13.7",
        "torch": "2.13.0+cpu",
    }
    for relative_path, expected in seal["source_sha256"].items():
        assert sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest() == expected
