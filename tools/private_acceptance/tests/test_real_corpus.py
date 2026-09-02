# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tools.private_acceptance.real_corpus import (
    AxisAnchor,
    assign_splits,
    axis_pixel_error,
    export_y_accuracy,
    inventory,
    marker_center_precision_recall,
    parse_dig,
)


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000c49444154789c6360f8cfc000000301010018dd8db40000000049454e44ae426082"
    )
).decode("ascii")


def _write(path: Path, study: str, index: int = 0, anchors: int = 3) -> None:
    anchor_xml = "".join(
        f'<Point><PositionScreen x="{10 + i}" y="{20 + i}"/><PositionGraph x="{i}" y="{i * 5}"/></Point>'
        for i in range(anchors)
    )
    points = '<Curve><Point><PositionScreen x="11" y="21"/></Point><Point><PositionScreen x="12" y="22"/></Point></Curve>'
    xml = f'<!DOCTYPE engauge><Document><SourceImage format="PNG">{PNG_1X1}</SourceImage><Axes>{anchor_xml}</Axes>{points}</Document>'
    target = path / study / f"graph-{index}.dig"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(xml, encoding="utf-8")


def test_parse_dig_retains_only_metadata_and_geometry(tmp_path: Path) -> None:
    _write(tmp_path, "Study A")
    target = tmp_path / "Study A" / "graph-0.dig"
    truth = parse_dig(target)
    assert truth.image.width == 1 and truth.image.height == 1
    assert truth.image.format == "PNG"
    assert len(truth.anchors) == 3
    assert len(truth.points) == 2
    payload = json.dumps(truth.__dict__, default=lambda value: value.__dict__)
    assert PNG_1X1 not in payload
    assert "Study A" not in payload


def test_three_anchor_validation(tmp_path: Path) -> None:
    _write(tmp_path, "Study", anchors=2)
    with pytest.raises(ValueError, match="DIG_AXIS_ANCHOR_COUNT:2"):
        parse_dig(tmp_path / "Study" / "graph-0.dig")


def test_external_entity_declaration_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "unsafe.dig"
    target.write_text(
        '<!DOCTYPE engauge SYSTEM "file:///private"><Document/>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DIG_EXTERNAL_ENTITY_FORBIDDEN"):
        parse_dig(target)


def test_declared_dimensions_do_not_mask_invalid_embedded_image(tmp_path: Path) -> None:
    target = tmp_path / "invalid-image.dig"
    target.write_text(
        '<Document><Image Width="1" Height="1">not-base64</Image>'
        '<Axes>'
        '<Point><PositionScreen X="0" Y="0"/><PositionGraph X="0" Y="0"/></Point>'
        '<Point><PositionScreen X="0" Y="1"/><PositionGraph X="0" Y="1"/></Point>'
        '<Point><PositionScreen X="1" Y="0"/><PositionGraph X="1" Y="0"/></Point>'
        '</Axes></Document>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DIG_EMBEDDED_IMAGE_INVALID"):
        parse_dig(target)


def test_declared_dimensions_do_not_replace_missing_embedded_image(tmp_path: Path) -> None:
    target = tmp_path / "missing-image.dig"
    target.write_text(
        '<Document><Image Width="1" Height="1"/>'
        '<Axes>'
        '<Point><PositionScreen X="0" Y="0"/><PositionGraph X="0" Y="0"/></Point>'
        '<Point><PositionScreen X="0" Y="1"/><PositionGraph X="0" Y="1"/></Point>'
        '<Point><PositionScreen X="1" Y="0"/><PositionGraph X="1" Y="0"/></Point>'
        '</Axes></Document>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DIG_EMBEDDED_IMAGE_MISSING"):
        parse_dig(target)


def test_study_isolation_and_determinism(tmp_path: Path) -> None:
    for study, count in (("A", 20), ("B", 20), ("C", 20), ("D", 20)):
        for index in range(count):
            _write(tmp_path, study, index)
    paths = sorted(tmp_path.rglob("*.dig"))
    first = assign_splits(paths, tmp_path, sealed_target=21)
    second = assign_splits(reversed(paths), tmp_path, sealed_target=21)
    assert first == second
    for study in ("A", "B", "C", "D"):
        values = {first[path] for path in paths if path.parent.name == study}
        assert len(values) == 1
    result = inventory(tmp_path, sealed_target=21)
    assert result.project_count == 80
    assert result.real_sealed_count in {20, 40}
    assert result.study_directory_count == 4
    assert result.study_count_with_projects == 4
    assert result.axis_anchor_count == 240
    assert result.digitized_point_count == 160
    assert result.as_dict()["git_eligible"] is False
    assert result.as_dict()["case_level_output"] is False
    serialized = json.dumps(result.as_dict(), sort_keys=True)
    assert "Study" not in serialized
    assert str(tmp_path) not in serialized
    assert result.assignment_sha256 == inventory(tmp_path, sealed_target=21).assignment_sha256


def test_metric_helpers() -> None:
    truth = (AxisAnchor(0, 0, 0, 0), AxisAnchor(3, 4, 1, 1), AxisAnchor(6, 8, 2, 2))
    predicted = (AxisAnchor(6, 8, 2, 2), AxisAnchor(3, 5, 1, 1), AxisAnchor(0, 0, 0, 0))
    assert axis_pixel_error(predicted, truth)["maximum_error_px"] == 1.0
    centers = marker_center_precision_recall([(0, 0), (10, 10), (99, 99)], [(0, 0), (10.5, 10)], 1)
    assert centers["true_positives"] == 2 and centers["false_positives"] == 1
    optimal = marker_center_precision_recall([(0.9, 0), (0, 0)], [(0, 0), (1.8, 0)], 1)
    assert optimal["true_positives"] == 2
    values = export_y_accuracy([(1.2, 10), (2.4, 21), (7, 5)], [(1, 14), (2, 25), (7, 12)])
    assert values["matched"] == 3 and values["within_tolerance"] == 2
    reordered = export_y_accuracy([(1, 5), (1, 0)], [(1, 0), (1, 10)])
    assert reordered["within_tolerance"] == 2


def test_cli_requires_explicit_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.private_acceptance import real_corpus

    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(SystemExit, match="PRIVATE_CORPUS_EXPLICIT_OPT_IN_REQUIRED"):
        real_corpus.main([str(tmp_path)])
    monkeypatch.setenv("CI", "true")
    with pytest.raises(SystemExit, match="PRIVATE_CORPUS_DISABLED_IN_CI"):
        real_corpus.main([str(tmp_path), "--explicit-opt-in"])
