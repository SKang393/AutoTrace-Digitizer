# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import base64
from io import BytesIO
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tools.private_acceptance import real_corpus, real_dev_distribution as probe
from tools.private_acceptance.real_corpus import assign_splits


def _write_dig(root: Path, study: str, index: int) -> Path:
    image = Image.new("RGB", (64, 48), "white")
    draw = ImageDraw.Draw(image)
    for x, y in ((16, 20), (40, 30)):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline="black", width=2)
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    payload = base64.b64encode(encoded.getvalue()).decode("ascii")
    anchors = "".join(
        f'<Point><PositionScreen X="{x}" Y="{y}"/><PositionGraph X="{i}" Y="{i * 5}"/></Point>'
        for i, (x, y) in enumerate(((5, 5), (5, 40), (55, 5)))
    )
    points = '<Curve><Point><PositionScreen X="16" Y="20"/></Point><Point><PositionScreen X="40" Y="30"/></Point></Curve>'
    path = root / study / f"graph-{index}.dig"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'<Document><SourceImage>{payload}</SourceImage><Axes>{anchors}</Axes>{points}</Document>', encoding="utf-8")
    return path


def test_diagnose_reads_only_real_dev_and_is_private_aggregate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [_write_dig(tmp_path, study, 0) for study in ("Study A", "Study B", "Study C", "Study D")]
    assignments = assign_splits(paths, tmp_path, sealed_target=2)
    sealed = {path for path, split in assignments.items() if split == "real-sealed"}
    loaded: list[Path] = []
    original = probe._load_embedded_png

    def tracked(path: Path) -> Image.Image:
        loaded.append(path)
        return original(path)

    monkeypatch.setattr(probe, "_load_embedded_png", tracked)
    before = {path: path.read_bytes() for path in paths}
    result = probe.diagnose(tmp_path, sealed_target=2)
    assert result["real_sealed_reads"] == 0
    assert result["real_dev"] == len(paths) - len(sealed)
    assert set(loaded).isdisjoint(sealed)
    assert result["source_dimensions"] == {"width_px": [64, 64, 64], "height_px": [48, 48, 48]}
    assert result["measured_marker_count"] == result["real_dev"] * 2
    assert result["truth_marker_count"] == result["measured_marker_count"]
    assert result["measurement_coverage"] == 1.0
    assert 7 <= result["effective_marker_diameter_px"]["median"] <= 10
    diameter = result["effective_marker_diameter_px"]
    assert diameter["p05"] <= diameter["p10"] <= diameter["median"] <= diameter["p90"] <= diameter["p95"]
    patch = result["marker_proposal_patch"]
    assert patch["width_px"] == 33
    assert patch["height_px"] == 33
    assert patch["diameter_to_patch_ratio"]["median"] == diameter["median"] / 33
    assert patch["diameter_to_patch_ratio"]["p95"] >= patch["diameter_to_patch_ratio"]["p90"]
    assert "identity 33x33" in patch["coverage_definition"]
    coverage = result["anchor_plot_point_coverage"]
    assert coverage["truth_marker_count"] == result["truth_marker_count"]
    assert coverage["inside_image_count"] == result["truth_marker_count"]
    assert coverage["inside_anchor_plot_count"] == result["truth_marker_count"]
    assert coverage["inside_expanded_proposal_search_count"] == result["truth_marker_count"]
    assert coverage["expanded_proposal_search_margin_px"] == 21
    serialized = json.dumps(result, sort_keys=True)
    assert "Study" not in serialized
    assert str(tmp_path) not in serialized
    assert {path: path.read_bytes() for path in paths} == before
    assert result == probe.diagnose(tmp_path, sealed_target=2)


def test_cli_requires_opt_in_and_rejects_ci(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(SystemExit, match="PRIVATE_CORPUS_EXPLICIT_OPT_IN_REQUIRED"):
        probe.main([str(tmp_path)])
    monkeypatch.setenv("CI", "true")
    with pytest.raises(SystemExit, match="PRIVATE_CORPUS_DISABLED_IN_CI"):
        probe.main([str(tmp_path), "--explicit-opt-in"])
