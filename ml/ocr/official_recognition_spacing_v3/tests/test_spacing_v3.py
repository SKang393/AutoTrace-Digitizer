# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.official_recognition_spacing_v3 import evaluate, prepare_split, sealed_gate
from ml.ocr.official_recognition_spacing_v3.protocol import (
    CANDIDATE_ID, GATES, MODEL_SHA256, REVISION, TRIGGER_REPORT_SHA256,
)
from ml.ocr.official_recognition_spacing_v3.spacing import restore_conservative_source_spaces


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/official_recognition_spacing_v3"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render_tokens(tokens: list[str], *, spacing: int) -> Image.Image:
    font = ImageFont.truetype(str(REPO_ROOT / "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"), 24)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bounds = [probe.textbbox((0, 0), token, font=font) for token in tokens]
    widths = [item[2] - item[0] for item in bounds]
    image = Image.new("RGB", (sum(widths) + spacing * max(0, len(tokens) - 1) + 12, 40), "white")
    draw = ImageDraw.Draw(image)
    x = 4.0
    for token, bound, width in zip(tokens, bounds, widths, strict=True):
        draw.text((x - bound[0], 2 - bound[1]), token, font=font, fill="black")
        x += width + spacing
    return image


def test_conservative_spacing_uses_source_pixels_without_semantics() -> None:
    source = _render_tokens(["O", "o", "l", "I"], spacing=14)
    assert restore_conservative_source_spaces(source, "OolI") == "O o l I"
    assert restore_conservative_source_spaces(source, "Oo lI") == "O o l I"
    assert restore_conservative_source_spaces(source, "WXYZ") == "W X Y Z"


def test_conservative_spacing_never_rewrites_nonspace_characters() -> None:
    source = _render_tokens(["O", "o", "l", "I"], spacing=14)
    for raw in ("OolI", "Ooll", "O o I I", "WXYl"):
        prediction = restore_conservative_source_spaces(source, raw)
        assert "".join(prediction.split()) == "".join(raw.split())


def test_conservative_spacing_does_not_split_compact_words() -> None:
    for text in ("Maintenance", "Baseline", "Intervention", "Target", "100", "10.0"):
        assert restore_conservative_source_spaces(_render_tokens([text], spacing=0), text) == text


def test_stronger_gap_threshold_ignores_incidental_internal_blank_band() -> None:
    image = Image.new("RGB", (24, 20), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, 7, 17), fill="black")
    draw.rectangle((14, 2, 19, 17), fill="black")
    assert restore_conservative_source_spaces(image, "AB") == "AB"


def test_fresh_split_bytes_reproduce_and_remain_private() -> None:
    for partition, seal_name in (("selection", "SELECTION_SEAL.json"), ("sealed_public", "SEALED_PUBLIC_TEST_SEAL.json")):
        manifest, archive = prepare_split.build_partition(partition)
        seal = _load(ROOT / seal_name)
        parsed = json.loads(manifest)
        assert prepare_split.hash_bytes(manifest) == seal["private_manifest_sha256"]
        assert prepare_split.hash_bytes(archive) == seal["fixture_archive_sha256"]
        assert parsed["synthetic_only"] is True
        assert parsed["private_or_article_images"] is False
        assert parsed["chandler_included"] is False
        assert parsed["generalization_label_included"] is False


def test_preregistration_binds_exact_failure_payload_sources_and_unopened_gate() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    config = _load(ROOT / "training/p1.json")
    selection = _load(ROOT / "SELECTION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    ledger = _load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert protocol["runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, evaluate.RUNNER_SOURCE_PATHS)
    assert config["expected_runner_source_bundle_sha256"] == protocol["runner_source_bundle_sha256"]
    assert config["model_sha256"] == MODEL_SHA256 == sha256_file(REPO_ROOT / evaluate.MODEL_PATH)
    assert config["trigger_report_sha256"] == TRIGGER_REPORT_SHA256 == sha256_file(REPO_ROOT / evaluate.TRIGGER_REPORT_PATH)
    assert config["protocol_sha256"] == entry["p1_preregistration_protocol_sha256"]
    assert config["protocol_sha256"] != sha256_file(ROOT / "PROTOCOL.json")
    assert config["selection_seal_sha256"] == sha256_file(ROOT / "SELECTION_SEAL.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert selection["model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["public_gate_evaluations"] == 0
    assert CANDIDATE_ID == "P1"
    assert entry["status"] == "candidate_1_failed_selection"
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_archive_opened"] is False
    assert entry["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert protocol["gates"] == GATES
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_public_gate_binding_requires_committed_selection_authorization() -> None:
    config = _load(ROOT / "gates/sealed-public-p1.json")
    ledger = _load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert config["expected_evaluator_source_bundle_sha256"] == entry["p1_preregistered_public_evaluator_source_bundle_sha256"]
    assert config["expected_evaluator_source_bundle_sha256"] != source_bundle_sha256(
        REPO_ROOT, sealed_gate.EVALUATOR_SOURCE_PATHS
    )
    assert config["evaluation_limit"] == 1
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_archive_opened"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False


def test_no_manifest_or_model_store_promotion_exists() -> None:
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*conservative*spacing*v3*.json"))
    index = _load(REPO_ROOT / "artifacts/production-model-store/production-model-index.json")
    assert REVISION not in json.dumps(index, sort_keys=True)
