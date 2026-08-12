# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.official_recognition_spacing_v2 import evaluate, prepare_split, sealed_gate
from ml.ocr.official_recognition_spacing_v2.protocol import CANDIDATE_ID, GATES, MODEL_SHA256, REVISION
from ml.ocr.official_recognition_spacing_v2.spacing import (
    restore_source_evidenced_spaces,
    restore_source_evidenced_spaces_and_vertical_case,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/official_recognition_spacing_v2"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _render(text: str, *, spacing: int = 0) -> Image.Image:
    font = ImageFont.truetype(str(REPO_ROOT / "src/GraphReader.App/Assets/Fonts/NotoSans-Regular.ttf"), 24)
    if spacing == 0:
        image = Image.new("RGB", (180, 40), "white")
        ImageDraw.Draw(image).text((4, 2), text, font=font, fill="black")
        return image.crop(image.getbbox())
    glyphs = [character for character in text]
    widths = [ImageDraw.Draw(Image.new("RGB", (1, 1))).textlength(glyph, font=font) for glyph in glyphs]
    image = Image.new("RGB", (int(sum(widths) + spacing * (len(glyphs) - 1) + 12), 40), "white")
    draw = ImageDraw.Draw(image)
    x = 4.0
    for glyph, width in zip(glyphs, widths, strict=True):
        draw.text((x, 2), glyph, font=font, fill="black")
        x += width + spacing
    return image


def test_generic_spacing_rule_uses_pixels_not_truth_or_role() -> None:
    image = _render("OolI", spacing=8)
    assert restore_source_evidenced_spaces(image, "OolI") == "O o l I"
    assert restore_source_evidenced_spaces(image, "WXYZ") == "W X Y Z"
    assert restore_source_evidenced_spaces(image, "O o l I") == "O o l I"


def test_spacing_rule_does_not_change_compact_nonspace_text() -> None:
    assert restore_source_evidenced_spaces(_render("Chandler"), "Chandler") == "Chandler"
    assert restore_source_evidenced_spaces(_render("100"), "100") == "100"
    assert restore_source_evidenced_spaces(_render("10.0"), "10.0") == "10.0"


def test_p2_vertical_case_rule_uses_source_serifs_without_truth_or_role() -> None:
    source = _render("OolI", spacing=8)
    assert restore_source_evidenced_spaces_and_vertical_case(source, "Ooll") == "O o l I"
    assert restore_source_evidenced_spaces_and_vertical_case(source, "WXYl") == "W X Y I"
    lowercase = _render("llll", spacing=8)
    assert restore_source_evidenced_spaces_and_vertical_case(lowercase, "llll") == "l l l l"
    assert restore_source_evidenced_spaces_and_vertical_case(_render("100"), "100") == "100"


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


def test_preregistration_binds_exact_weights_sources_and_unopened_public_gate() -> None:
    protocol = _load(ROOT / "PROTOCOL.json")
    config = _load(ROOT / "training/p2.json")
    selection = _load(ROOT / "SELECTION_SEAL.json")
    public = _load(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    gate = _load(ROOT / "gates/sealed-public-p2.json")
    ledger = _load(REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json")
    entry = next(item for item in ledger["revisions"] if item["revision"] == REVISION)
    assert protocol["runner_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, evaluate.RUNNER_SOURCE_PATHS)
    assert config["expected_runner_source_bundle_sha256"] == protocol["runner_source_bundle_sha256"]
    assert config["model_sha256"] == MODEL_SHA256 == sha256_file(REPO_ROOT / evaluate.MODEL_PATH)
    assert config["protocol_sha256"] == sha256_file(ROOT / "PROTOCOL.json")
    assert config["p1_result_sha256"] == sha256_file(ROOT / "P1_RESULT.json")
    assert config["selection_seal_sha256"] == sha256_file(ROOT / "SELECTION_SEAL.json")
    assert config["sealed_public_test_seal_sha256"] == sha256_file(ROOT / "SEALED_PUBLIC_TEST_SEAL.json")
    assert gate["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(REPO_ROOT, sealed_gate.EVALUATOR_SOURCE_PATHS)
    assert selection["model_execution_count"] == 0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["public_gate_evaluations"] == 0
    assert CANDIDATE_ID == "P2"
    assert entry["status"] == "candidate_2_preregistered"
    assert entry["execution_authorized"] is True
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_archive_opened"] is False
    assert protocol["gates"] == GATES
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_no_manifest_or_model_store_promotion_exists() -> None:
    assert not any(REPO_ROOT.glob("models/manifest/ocr/*spacing*v2*.json"))
    index = _load(REPO_ROOT / "artifacts/production-model-store/production-model-index.json")
    assert REVISION not in json.dumps(index, sort_keys=True)
