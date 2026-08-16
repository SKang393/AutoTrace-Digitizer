# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml.markers.gate_seal import GateSeal
from ml.ocr.official_bakeoff import structure_consensus_evaluate as base
from ml.ocr.official_bakeoff import structure_consensus_v2_evaluate as gate


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PROTOCOL = Path(gate.__file__).with_name("STRUCTURE_CONSENSUS_V2_GATE_PROTOCOL.json")
METRICS = REPOSITORY_ROOT / "ml" / "ocr" / "production_gate.py"


class _Metadata:
    def __init__(self, name: str) -> None:
        self.name = name


class _Session:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[_Metadata]:
        return [_Metadata("x")]

    def get_outputs(self) -> list[_Metadata]:
        return [_Metadata("sigmoid_0.tmp_0")]

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert output_names == ["sigmoid_0.tmp_0"]
        assert list(inputs) == ["x"]
        return [self.output]


def _output() -> np.ndarray:
    return np.zeros((1, 1, 512, 1024), dtype=np.float32)


def _image_bytes() -> bytes:
    return bytes([255]) * (320 * 160 * 3)


def test_protocol_binds_distinct_sources_activation_disjointness_and_one_run() -> None:
    protocol = gate.validate_protocol(PROTOCOL, METRICS)

    assert protocol["profile"] == gate.PROFILE
    assert protocol["defect_class"] == "bounded_probability_runtime_activation"
    assert protocol["candidate"]["output_activation"] == gate.OUTPUT_ACTIVATION
    assert protocol["candidate"]["probability_tolerance"] == gate.PROBABILITY_TOLERANCE
    assert len(protocol["prior_exposed_splits_forbidden"]) == 2
    assert protocol["new_split"]["render_index_offset"] == gate.RENDER_INDEX_OFFSET
    assert protocol["evaluation_authorization"]["config_must_be_committed_after_single_freeze"] is True
    assert protocol["experiment_budget"]["official_composition_evaluations"] == 1
    assert protocol["private_data"] is False
    assert protocol["chandler_used"] is False


def test_bounded_activation_clamps_only_finite_drift_and_hashes_raw_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base, "db_model_regions", lambda *_args: ())
    monkeypatch.setattr(base, "connected_component_candidates", lambda *_args: ())
    monkeypatch.setattr(base, "compose_consensus", lambda *_args: ((), ()))
    output = _output()
    output[0, 0, 0, 0] = np.float32(1.000005)
    output[0, 0, 0, 1] = np.float32(-0.000005)
    raw_sha256 = gate.hash_bytes(output.tobytes(order="C"))
    gate._detector_observations.clear()

    evidence = gate.detect_regions(_Session(output), _image_bytes(), 320, 160)

    assert evidence.output_sha256 == raw_sha256
    summary = gate._activation_summary()
    assert summary["clamped_value_count"] == 2
    assert summary["maximum_boundary_drift"] <= gate.PROBABILITY_TOLERANCE
    assert summary["material_drift_rejected"] is True
    assert summary["non_finite_rejected"] is True


@pytest.mark.parametrize("invalid", [np.float32(1.00002), np.float32(-0.00002)])
def test_bounded_activation_rejects_material_drift(invalid: np.float32) -> None:
    output = _output()
    output[0, 0, 0, 0] = invalid

    with pytest.raises(gate.ProductionGateError, match="fixed 1e-5 probability tolerance"):
        gate.detect_regions(_Session(output), _image_bytes(), 320, 160)


def test_bounded_activation_rejects_non_finite_output() -> None:
    output = _output()
    output[0, 0, 0, 0] = np.float32(np.nan)

    with pytest.raises(gate.ProductionGateError, match="non-finite"):
        gate.detect_regions(_Session(output), _image_bytes(), 320, 160)


def test_configured_base_restores_consumed_v1_module() -> None:
    original_profile = base.PROFILE
    original_file = base.__file__
    original_detector = base.detect_regions

    with gate._configured_base():
        assert base.PROFILE == gate.PROFILE
        assert Path(base.__file__).resolve() == Path(gate.__file__).resolve()
        assert base.detect_regions is gate.detect_regions

    assert base.PROFILE == original_profile
    assert base.__file__ == original_file
    assert base.detect_regions is original_detector


def test_activation_binding_rehashes_every_dependent_resource(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "core-predictions.json").write_bytes(gate.canonical_json_bytes({"schema": gate.CORE_SCHEMA}))
    (output / "predictions.json").write_bytes(
        gate.canonical_json_bytes({"schema": gate.PREDICTIONS_SCHEMA, "core_predictions_sha256": "old"})
    )
    (output / "runtime-results.json").write_bytes(
        gate.canonical_json_bytes(
            {
                "schema": gate.RUNTIME_SCHEMA,
                "core_predictions_sha256": "old",
                "predictions_sha256": "old",
                "execution_provenance": {},
            }
        )
    )
    (output / "report.json").write_bytes(
        gate.canonical_json_bytes(
            {
                "schema": gate.REPORT_SCHEMA,
                "status": "fail",
                "production_approval": False,
                "reviewed_resources": {},
            }
        )
    )
    seal_directory = tmp_path / "seal"
    seal_directory.mkdir()
    opened = seal_directory / "opened.json"
    opened.write_bytes(gate.canonical_json_bytes({"status": "opened"}))
    seal = GateSeal("seal-key", seal_directory, opened, {"revision": gate.GATE_REVISION})
    gate._detector_observations.clear()
    gate._detector_observations.append(
        {
            "observed_minimum": -0.000005,
            "observed_maximum": 1.000005,
            "maximum_boundary_drift": 0.000005,
            "clamped_value_count": 2,
        }
    )

    report = gate._bind_activation_evidence(output, seal)

    core_sha256 = gate.hash_file(output / "core-predictions.json")
    predictions_sha256 = gate.hash_file(output / "predictions.json")
    runtime_sha256 = gate.hash_file(output / "runtime-results.json")
    predictions = gate.load_strict_json(output / "predictions.json")
    runtime = gate.load_strict_json(output / "runtime-results.json")
    assert predictions["core_predictions_sha256"] == core_sha256
    assert runtime["core_predictions_sha256"] == core_sha256
    assert runtime["predictions_sha256"] == predictions_sha256
    assert report["runtime_results_sha256"] == runtime_sha256
    assert report["detector_output_activation"]["clamped_value_count"] == 2
    assert report["public_gate_seal"]["key"] == "seal-key"
    assert report["production_approval"] is False
    assert set(report["reviewed_resources"]) == {
        "core_predictions",
        "predictions",
        "runtime_results",
    }


def test_evaluation_cli_requires_authorization_failure_record_and_prior_roots() -> None:
    with pytest.raises(SystemExit):
        gate.parse_args(
            [
                "evaluate",
                "--frozen-root", "frozen",
                "--conversion-report", "conversion.json",
                "--source-root", "source",
                "--output-root", "output",
            ]
        )
