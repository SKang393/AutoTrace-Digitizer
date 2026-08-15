# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from ml.ocr.production_csharp_marker_gate_v4.dataset import build_scene
from ml.ocr.production_csharp_marker_gate_v4.protocol import (
    DEGRADATION_FAMILIES,
    MODEL_SHA256,
    RENDERER_FAMILIES,
    SCENE_COUNT,
    protocol_configuration,
)
from ml.ocr.production_csharp_marker_gate_v2.protocol import (
    DEGRADATION_FAMILIES as V2_DEGRADATION_FAMILIES,
    MODEL_SHA256 as V2_MODEL_SHA256,
    RENDERER_FAMILIES as V2_RENDERER_FAMILIES,
)
from ml.ocr.production_csharp_marker_gate_v3.protocol import (
    DEGRADATION_FAMILIES as V3_DEGRADATION_FAMILIES,
    MODEL_SHA256 as V3_MODEL_SHA256,
    RENDERER_FAMILIES as V3_RENDERER_FAMILIES,
)
from ml.ocr.production_composition_v8.protocol import SPLITS as V8_SPLITS
from ml.ocr.component_context_detector_v7.dataset import box_iou, proposals


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_protocol_is_frozen_fail_closed_and_private_free() -> None:
    protocol = protocol_configuration()
    tracked = json.loads(
        (REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v4/PROTOCOL.json").read_text(encoding="utf-8")
    )
    assert protocol["schema"] == tracked["schema"]
    assert protocol["revision"] == tracked["revision"]
    assert protocol["composition_id"] == tracked["composition_id"]
    assert protocol["state"] == "frozen_before_sealed_public_identity_materialization_or_model_execution"
    assert protocol["split"]["scene_count"] == tracked["split"]["scene_count"] == SCENE_COUNT
    assert protocol["synthetic_only"] is True
    assert protocol["private_or_article_images"] is False
    assert protocol["chandler_included"] is False
    assert protocol["generalization_label_included"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert protocol["manifest_created"] is False
    assert protocol["model_store_promoted"] is False


def test_exact_v8_models_and_direct_evidence_requirements_are_frozen() -> None:
    protocol = protocol_configuration()
    assert len(MODEL_SHA256) == 5
    assert MODEL_SHA256["ocr_detector"] == (
        "474b8468dbd91416f4e4978dafc46cb2317775d59d821c0470e0cd3e0f6203db"
    )
    assert MODEL_SHA256["ocr_detector"] == V2_MODEL_SHA256["ocr_detector"]
    assert MODEL_SHA256["ocr_detector"] != V3_MODEL_SHA256["ocr_detector"]
    assert all(len(value) == 64 and int(value, 16) >= 0 for value in MODEL_SHA256.values())
    assert {item["sha256"] for item in protocol["models"].values()} == set(MODEL_SHA256.values())
    assert {item["provider"] for item in protocol["models"].values()} == {"CPUExecutionProvider"}
    gates = protocol["gates"]
    assert gates["direct_fixture_byte_execution_required"] is True
    assert gates["direct_tensor_hash_per_model_call_required"] is True
    assert gates["single_authorized_execution"] is True
    assert gates["exact_ocr_region_count_every_fixture"] is True
    assert gates["exact_marker_count_every_fixture"] is True
    assert gates["text_marker_creation_count"] == 0
    assert gates["line_intersection_hits"] == 0
    assert gates["every_required_role_observed"] is True
    assert gates["full_eight_role_coverage_proven"] is False


def test_composite_families_are_disjoint_from_predecessor_gates_and_v8_splits() -> None:
    protocol = protocol_configuration()
    v8_renderers = {item.renderer_family for item in V8_SPLITS}
    v8_degradations = {item.degradation_family for item in V8_SPLITS}
    assert len(RENDERER_FAMILIES) == 8 and len(set(RENDERER_FAMILIES)) == 8
    assert len(DEGRADATION_FAMILIES) == 4 and len(set(DEGRADATION_FAMILIES)) == 4
    assert set(RENDERER_FAMILIES).isdisjoint(
        set(V2_RENDERER_FAMILIES) | set(V3_RENDERER_FAMILIES) | v8_renderers
    )
    assert set(DEGRADATION_FAMILIES).isdisjoint(
        set(V2_DEGRADATION_FAMILIES) | set(V3_DEGRADATION_FAMILIES) | v8_degradations
    )
    assert protocol["split"]["predecessor_fixture_bytes_reused"] is False
    assert protocol["split"]["predecessor_truth_or_scene_ids_reused"] is False
    assert protocol["split"]["secret_seed_generated_once_and_not_serialized"] is True
    assert protocol["composition"]["artifact_mask_production_approval"] is False
    assert "approved_artifact_mask_provider" in protocol["blocking_gates_after_pass"]
    assert "full_eight_role_coverage" in protocol["blocking_gates_after_pass"]


def test_scene_has_all_five_preregistered_roles_and_real_marker_structure_truth() -> None:
    scene = build_scene(0x9BD5A0173C42EF11, 0)
    assert scene.scene_id.startswith("ocr-marker-csharp-v4-public-")
    assert scene.raster.shape == (320, 640)
    assert scene.raster.dtype.name == "uint8"
    assert scene.artifact_mask.shape == scene.raster.shape
    assert set(scene.artifact_mask.ravel()).issubset({0, 255})
    assert len(scene.text_truths) == 5
    assert {item.role for item in scene.text_truths} == {
        "y_tick", "x_tick", "phase_heading", "annotation", "legend_text",
    }
    assert len(scene.marker_centers) >= 8
    assert {item.family for item in scene.text_truths} >= {"numeric", "word"}
    assert any(
        truth.family == "ambiguity"
        for index in range(SCENE_COUNT)
        for truth in build_scene(0x9BD5A0173C42EF11, index).text_truths
    )
    assert {item.kind for item in scene.prohibited} >= {
        "axis", "tick", "divider", "legend", "bracket", "arrowhead",
    }
    combined = " ".join(item.display_text for item in scene.text_truths)
    assert "Chandler" not in combined
    assert "Generalization" not in combined
    assert all(0 <= x < 640 and 0 <= y < 320 for x, y in scene.marker_centers)


def test_visible_design_seed_has_one_static_proposal_for_every_truth() -> None:
    visible_design_seed = 0x7A33C1540F2D9911
    for index in range(SCENE_COUNT):
        scene = build_scene(visible_design_seed, index)
        candidates = proposals(scene.raster)
        for truth in scene.text_truths:
            assert sum(box_iou(candidate.box, truth.box) >= 0.5 for candidate in candidates) == 1


def test_preregistration_has_no_split_or_execution_result() -> None:
    root = REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v4"
    seal_exists = (root / "SEALED_PUBLIC_TEST_SEAL.json").exists()
    if not seal_exists:
        assert not (REPO_ROOT / "artifacts/production-validation/ocr-marker-csharp-v4-sealed.zip").exists()
    assert not (root / "PUBLIC_GATE_RESULT.json").exists()


def test_future_split_seal_is_hash_only_fail_closed_and_has_no_secret() -> None:
    seal_path = REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v4/SEALED_PUBLIC_TEST_SEAL.json"
    if not seal_path.exists():
        return
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert seal["schema"] == "graphreader.ocr-marker-production-composition-split-seal.v1"
    assert seal["scene_count"] == SCENE_COUNT
    assert seal["text_truth_count"] == SCENE_COUNT * 5
    assert seal["model_execution_count_at_freeze"] == 0
    assert seal["secret_seed_serialized"] is False
    assert "secret_seed" not in seal
    assert seal["production_approval"] is False
    assert seal["release_eligible"] is False
    for relative_path, expected in seal["source_sha256"].items():
        assert sha256((REPO_ROOT / relative_path).read_bytes()).hexdigest() == expected


def test_public_gate_authorization_binds_exact_sealed_identity_and_candidates() -> None:
    authorization_path = REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v4/PUBLIC_GATE_AUTHORIZATION.json"
    if not authorization_path.exists():
        return
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    seal_path = REPO_ROOT / "ml/ocr/production_csharp_marker_gate_v4/SEALED_PUBLIC_TEST_SEAL.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert authorization["execution_authorized"] is True
    assert authorization["execution_count_authorized"] == 1
    assert authorization["provider"] == "CPUExecutionProvider"
    assert len(authorization["sealed_identity_commit"]) == 40
    assert int(authorization["sealed_identity_commit"], 16) >= 0
    assert authorization["fixture_archive_sha256"] == seal["fixture_archive_sha256"]
    assert authorization["fixture_manifest_sha256"] == seal["fixture_manifest_sha256"]
    assert authorization["split_seal_sha256"] == sha256(seal_path.read_bytes()).hexdigest()
    assert authorization["candidate_sha256"] == sorted(MODEL_SHA256.values())
    assert authorization["exact_test"] == (
        "OcrMarkerDirectPublicGateV4Tests.NewSealedBytesExecuteOnceThroughCSharpOcrAndMarkerCpuComposition"
    )
    assert authorization["result_path"] == "artifacts/production-validation/ocr-marker-csharp-v4-report.json"
    assert authorization["rerun_or_repair_authorized"] is False
    assert authorization["artifact_mask_production_approval"] is False
    assert authorization["manifest_creation_authorized"] is False
    assert authorization["model_store_promotion_authorized"] is False
    assert authorization["private_validation_authorized"] is False
    assert authorization["production_approval"] is False
    assert authorization["release_eligible"] is False
