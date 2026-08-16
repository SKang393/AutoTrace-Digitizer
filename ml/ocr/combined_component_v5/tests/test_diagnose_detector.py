# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ml.ocr.combined_component_v5 import diagnose_detector as diagnostic


ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = Path(diagnostic.__file__).with_name("DETECTOR_DIAGNOSTIC_PROTOCOL.json")
FONT = ROOT / "src" / "GraphReader.App" / "Assets" / "Fonts" / "NotoSans-Regular.ttf"


class FakeSession:
    def __init__(self, minimum: float = 0.0, maximum: float = 1.0) -> None:
        self.minimum = minimum
        self.maximum = maximum

    @staticmethod
    def get_inputs() -> list[SimpleNamespace]:
        return [SimpleNamespace(name="x")]

    @staticmethod
    def get_outputs() -> list[SimpleNamespace]:
        return [SimpleNamespace(name="y")]

    def run(self, _outputs: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        tensor = next(iter(feeds.values()))
        output = np.zeros((1, 1, tensor.shape[2], tensor.shape[3]), dtype=np.float32)
        output.flat[0] = np.float32(self.minimum)
        output.flat[-1] = np.float32(self.maximum)
        return [output]


def test_frozen_protocol_is_non_approval_and_source_bound() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False
    assert protocol["experiment_budget"]["detector_diagnostic_runs"] == 1
    with pytest.raises(
        diagnostic.DiagnosticError,
        match="Reviewed source changed: src/GraphReader.Ocr/LocalOnnxTextRegionDetector.cs",
    ):
        diagnostic.validate_protocol(PROTOCOL)


def test_renderer_is_deterministic_and_uses_reserved_dimensions() -> None:
    first = diagnostic.render_case(7, FONT)
    repeated = diagnostic.render_case(7, FONT)
    different = diagnostic.render_case(8, FONT)
    assert first.source_sha256 == repeated.source_sha256
    assert first.detector_bgr_sha256 == repeated.detector_bgr_sha256
    assert first.source_sha256 != different.source_sha256
    assert len(first.detector_bgr) == diagnostic.WIDTH * diagnostic.HEIGHT * 3


def test_out_of_range_probability_is_measured_without_approval() -> None:
    case = diagnostic.render_case(diagnostic.TEXT_CASES, FONT)
    record = diagnostic.diagnose_case(FakeSession(-1e-6, 1.000001), case)
    assert record["strict_probability"] is False
    assert record["diagnostic_clamp_applied_for_region_analysis"] is True
    assert record["raw_minimum"] < 0
    assert record["raw_maximum"] > 1


def test_run_refuses_to_overwrite_consumed_output(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["reviewed_source_sha256"] = {
        relative: diagnostic._hash_file(ROOT / relative)
        for relative in protocol["reviewed_source_sha256"]
    }
    current_protocol = tmp_path / "current-source-protocol.json"
    current_protocol.write_text(json.dumps(protocol), encoding="utf-8")
    output = tmp_path / "consumed"
    output.mkdir()
    with pytest.raises(diagnostic.DiagnosticError, match="overwrite"):
        diagnostic.run_diagnostic(
            current_protocol, tmp_path / "missing.onnx", output, FONT,
        )
