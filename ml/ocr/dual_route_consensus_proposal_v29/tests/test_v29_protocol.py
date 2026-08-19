# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
import inspect
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.ocr.dual_route_consensus_proposal_v29.model import (
    DualRouteConsensusProposalNet,
)
from ml.ocr.dual_route_consensus_proposal_v29.dataset import render_scene
from ml.ocr.dual_route_consensus_proposal_v29.prepare_split import (
    ARCHIVE_PATHS,
    SEAL_PATH,
    SOURCE_PATHS,
)
from ml.ocr.dual_route_consensus_proposal_v29.protocol import (
    REVISION,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.dual_route_consensus_proposal_v29.train_p1 import (
    _dual_route_objective,
    _trigger_is_terminal,
    preflight,
)
from ml.ocr.dual_route_consensus_proposal_v29.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    EXPECTED_CANDIDATE_HASH_KEYS,
    GATE_CONFIG,
    PUBLIC_CONFIG_PATH,
    _gate_metrics_pass,
    _public_window,
    _selected_result_is_terminal,
    evaluate_public,
)
from ml.markers.gate_seal import canonical_json_bytes, source_bundle_sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/dual_route_consensus_proposal_v29"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ledger_entry() -> dict[str, object]:
    ledger = _read_json(
        REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"
    )
    return next(
        item for item in ledger["revisions"]
        if item["revision"] == REVISION
    )


def test_protocol_json_is_exact_preregistration() -> None:
    canonical_protocol = json.loads(json.dumps(protocol_configuration()))
    assert _read_json(ROOT / "PROTOCOL.json") == canonical_protocol
    protocol = protocol_configuration()
    assert protocol["training_authorized"] is False
    assert protocol["public_execution_authorized"] is False
    assert protocol["private_validation_authorized"] is False
    assert protocol["production_approval"] is False
    assert protocol["release_eligible"] is False


def test_only_tracked_aggregate_v28_result_triggers_v29() -> None:
    trigger_path = REPO_ROOT / TRIGGER_RESULT_PATH
    assert _sha256(trigger_path) == TRIGGER_RESULT_SHA256
    trigger = _read_json(trigger_path)
    assert _trigger_is_terminal(trigger)
    assert trigger["case_level_failure_analysis_performed"] is False
    assert trigger["next_revision_may_reuse_public_bytes"] is False
    assert trigger["public_failure_tuning_authorized"] is False
    for prohibited in ("cases", "predictions", "truths", "fixture_bytes"):
        assert prohibited not in trigger


def test_fresh_split_registrations_are_pairwise_disjoint() -> None:
    registrations = [
        split_registration("train"),
        split_registration("validation"),
        split_registration("sealed_public"),
    ]
    assert [item.scene_count for item in registrations] == [320, 160, 224]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    for first in range(3):
        for second in range(first + 1, 3):
            assert renderer_sets[first].isdisjoint(renderer_sets[second])
            assert degradation_sets[first].isdisjoint(degradation_sets[second])
    assert all("v29" in name for values in renderer_sets for name in values)
    assert all("v29" in name for values in degradation_sets for name in values)


def test_fresh_renderers_are_deterministic_and_keep_truth_geometry() -> None:
    for split in ("train", "validation", "sealed_public"):
        first = render_scene(split, 0)
        repeated = render_scene(split, 0)
        assert first.scene_id == repeated.scene_id
        assert first.renderer_family == repeated.renderer_family
        assert first.degradation_family == repeated.degradation_family
        assert first.truths == repeated.truths
        assert first.raster.shape == repeated.raster.shape
        assert torch.equal(
            torch.from_numpy(first.raster), torch.from_numpy(repeated.raster),
        )
        assert len(first.truths) == 8


def test_dual_route_model_has_dynamic_shape_and_deterministic_roles() -> None:
    torch.manual_seed(2901)
    model = DualRouteConsensusProposalNet().eval()
    evidence = torch.zeros((1, 7, 31), dtype=torch.float32)
    evidence[0, :, 25] = torch.tensor((-0.2, 0.5, 0.5, 0.5, 1.2, 1.2, 0.5))
    evidence[0, :, 26] = torch.tensor((0.5, 1.05, 1.2, -0.2, 0.5, 1.2, 0.5))
    crops = torch.rand((1, 7, 2, 32, 128), dtype=torch.float32)
    relations = torch.rand((1, 7, 7, 19), dtype=torch.float32) * 2.0 - 1.0
    with torch.inference_mode():
        output = model(evidence, crops, relations)
    assert output.shape == (1, 7, 10)
    assert torch.argmax(output[0, :, 2:], dim=1).tolist() == list(range(7))
    assert not any(
        name.startswith("attention_route.role_parent.") and parameter.requires_grad
        for name, parameter in model.named_parameters()
    )


def test_model_is_permutation_equivariant_over_proposals() -> None:
    torch.manual_seed(2902)
    model = DualRouteConsensusProposalNet().eval()
    evidence = torch.rand((1, 5, 31), dtype=torch.float32)
    crops = torch.rand((1, 5, 2, 32, 128), dtype=torch.float32)
    relations = torch.rand((1, 5, 5, 19), dtype=torch.float32) * 2.0 - 1.0
    order = torch.tensor((3, 0, 4, 1, 2))
    with torch.inference_mode():
        original = model(evidence, crops, relations)
        permuted = model(
            evidence[:, order],
            crops[:, order],
            relations[:, order][:, :, order],
        )
    torch.testing.assert_close(permuted, original[:, order], rtol=0.0, atol=2e-5)


def test_random_weight_onnx_contract_is_dynamic_and_within_strict_parity(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2903)
    model = DualRouteConsensusProposalNet().eval()
    evidence = torch.randn(1, 7, 31)
    crops = torch.randn(1, 7, 2, 32, 128)
    relations = torch.randn(1, 7, 7, 19)
    path = tmp_path / "v29-contract.onnx"
    torch.onnx.export(
        model,
        (evidence, crops, relations),
        path,
        input_names=["proposal_evidence", "proposal_crops", "proposal_relations"],
        output_names=["proposal_role_logits"],
        dynamic_axes={
            "proposal_evidence": {1: "proposal_count"},
            "proposal_crops": {1: "proposal_count"},
            "proposal_relations": {1: "proposal_count", 2: "neighbor_count"},
            "proposal_role_logits": {1: "proposal_count"},
        },
        opset_version=18,
        dynamo=False,
    )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        path.read_bytes(), sess_options=options, providers=["CPUExecutionProvider"],
    )
    for count in (4, 7):
        values = evidence[:, :count].numpy().astype(np.float32)
        crop_values = crops[:, :count].numpy().astype(np.float32)
        relation_values = relations[:, :count, :count].numpy().astype(np.float32)
        with torch.inference_mode():
            expected = model(
                torch.from_numpy(values),
                torch.from_numpy(crop_values),
                torch.from_numpy(relation_values),
            ).numpy()
        actual = session.run(None, {
            "proposal_evidence": values,
            "proposal_crops": crop_values,
            "proposal_relations": relation_values,
        })[0]
        assert actual.shape == (1, count, 10)
        assert float(np.max(np.abs(expected - actual))) <= 1e-5


def test_dual_route_objective_is_finite_and_differentiable() -> None:
    config = protocol_configuration()["candidate_p1"]
    consensus = torch.tensor(
        ((2.0, -2.0), (-2.0, 2.0), (1.0, -1.0), (-1.0, 1.0)),
        requires_grad=True,
    )
    attention = consensus.detach().clone().requires_grad_(True)
    summary = (consensus.detach() * 0.9).requires_grad_(True)
    targets = torch.tensor((0, 1, 0, 1), dtype=torch.long)
    weights = torch.ones(2, dtype=torch.float32)
    loss, parts = _dual_route_objective(
        consensus, attention, summary, targets, weights, config,
    )
    assert torch.isfinite(loss)
    assert set(("ensemble", "attention_route", "summary_route", "worst_route", "route_agreement")) <= set(parts)
    loss.backward()
    assert consensus.grad is not None
    assert attention.grad is not None
    assert summary.grad is not None


def test_source_only_ledger_is_fail_closed_before_fixture_freeze() -> None:
    entry = _ledger_entry()
    assert entry["status"] in {
        "preregistered_source_only",
        "candidate_1_preregistered",
        "candidate_1_selected",
        "candidate_1_failed_selection",
        "candidate_1_selected_public_gate_preregistered",
        "candidate_1_selected_public_gate_pending",
        "public_gate_failed_revision_closed",
    }
    assert entry["prior_fixture_bytes_reused"] is False
    assert entry["trigger_case_detail_or_pixels_used"] is False
    assert entry["public_gate_authorized"] is (
        entry["status"] == "candidate_1_selected_public_gate_pending"
    )
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    if entry["status"] == "preregistered_source_only":
        assert entry["split_materialized"] is False
        assert not (REPO_ROOT / SEAL_PATH).exists()
        assert all(not (REPO_ROOT / path).exists() for path in ARCHIVE_PATHS.values())


def test_frozen_p1_inputs_validate_before_separate_execution_authorization() -> None:
    entry = _ledger_entry()
    if entry["status"] != "candidate_1_preregistered":
        pytest.skip("P1 preregistration checkpoint has advanced")
    assert (REPO_ROOT / SEAL_PATH).is_file()
    assert all((REPO_ROOT / path).is_file() for path in ARCHIVE_PATHS.values())
    if entry["execution_authorized"] is False:
        with pytest.raises(RuntimeError, match="canonical authorization changed"):
            preflight()
    else:
        assert entry["execution_authorized"] is True
        assert entry["authorized_candidate_id"] == "P1"
        assert preflight()["seal"]["public_execution_authorized"] is False


def test_selected_p1_result_is_aggregate_only_and_fail_closed() -> None:
    entry = _ledger_entry()
    result_path = ROOT / "P1_RESULT.json"
    assert result_path.is_file()
    result = _read_json(result_path)
    assert result["schema"] == "graphreader.ocr-dual-route-consensus-selection-result.v1"
    assert result["status"] == "selected"
    assert result["candidate_consumed"] is True
    assert result["report_sha256"] == "49b7cd6d2645d7e5bda5c787a0bad9547cb2a244dc694c4ec77a17266672ee4b"
    assert result["checkpoint_sha256"] == "1dd8fc613815402fdad389f38652dbf75e35f8e25ed2f56dd06f06be6196336f"
    assert result["onnx_sha256"] == "a1ce725897f44d43a6db0852638abb3787c9be917bba0d412f0b1a798831f223"
    assert result["training_opened_seal_sha256"] == _sha256(
        REPO_ROOT / "ml/markers/training-seals/ocr-detection-recognition/graph-text-dual-route-consensus-proposal-v29/P1/opened.json"
    )
    assert result["training_result_seal_sha256"] == _sha256(
        REPO_ROOT / "ml/markers/training-seals/ocr-detection-recognition/graph-text-dual-route-consensus-proposal-v29/P1/result.json"
    )
    metrics = result["selection_metrics"]
    assert metrics["exact_scene_count"] == metrics["scene_count"] == 160
    assert metrics["true_positives"] == metrics["truth_region_count"] == 1280
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 0
    assert metrics["direct_stored_fixture_byte_execution"] is True
    assert result["passing_threshold_window"] == [0.35, 0.45, 0.55, 0.65, 0.75]
    assert result["onnx_parity_maximum_absolute_error"] <= 1e-5
    assert result["public_gate_evaluations"] == 0
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_authorized"] is False
    assert result["marker_creation_evaluated"] is False
    assert result["manifest_created"] is False
    assert result["model_store_promoted"] is False
    assert result["packaging_discovery"] is False
    assert result["private_validation_authorized"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False
    for prohibited in (
        "cases", "predictions", "truths", "fixture_bytes",
        "case_ids", "scene_ids", "proposal_relation_scene_shapes",
    ):
        assert prohibited not in result
        assert prohibited not in metrics
    assert entry["status"] in {
        "candidate_1_selected",
        "candidate_1_selected_public_gate_preregistered",
        "candidate_1_selected_public_gate_pending",
        "public_gate_failed_revision_closed",
    }
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["preregistered_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is (
        entry["status"] == "candidate_1_selected_public_gate_pending"
    )


def test_public_runner_is_consumed_with_aggregate_only_terminal_evidence() -> None:
    result = _read_json(ROOT / "P1_RESULT.json")
    assert _selected_result_is_terminal(result)
    assert _public_window(result) == (0.45, 0.55, 0.65)
    config_path = REPO_ROOT / PUBLIC_CONFIG_PATH
    config = _read_json(config_path)
    assert config["candidate_id"] == "P1"
    assert config["evaluation_limit"] == 1
    assert config["runner_source_commit"] == (
        "8667c6d5e915d6aac56f03012f852c9dd053fe3b"
    )
    assert config["expected_evaluator_source_bundle_sha256"] == source_bundle_sha256(
        REPO_ROOT, EVALUATOR_SOURCE_PATHS,
    )
    assert config["expected_gate_config_sha256"] == sha256(
        canonical_json_bytes(dict(GATE_CONFIG))
    ).hexdigest()
    assert config["expected_candidate_hash_keys"] == list(
        EXPECTED_CANDIDATE_HASH_KEYS
    )
    assert config["public_execution_authorized"] is True
    assert config["case_level_failure_analysis_permitted"] is False
    assert config["marker_creation_authorized"] is False
    assert config["private_validation_authorized"] is False
    assert config["production_approval"] is False
    assert config["release_eligible"] is False
    entry = _ledger_entry()
    assert entry["status"] == "public_gate_failed_revision_closed"
    assert entry["public_gate_config_path"] == PUBLIC_CONFIG_PATH.as_posix()
    assert entry["public_gate_config_sha256"] == _sha256(config_path)
    assert entry["public_gate_runner_source_commit"] == config["runner_source_commit"]
    assert entry["public_gate_runner_source_bundle_sha256"] == config[
        "expected_evaluator_source_bundle_sha256"
    ]
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_authorized_candidate_id"] is None
    assert entry["public_gate_evaluations"] == 1
    assert entry["public_gate_archive_opened"] is True
    public_result_path = ROOT / "PUBLIC_GATE_RESULT.json"
    public_result = _read_json(public_result_path)
    assert entry["public_gate_result_sha256"] == _sha256(public_result_path)
    assert public_result["gate_opened_seal_sha256"] == _sha256(
        REPO_ROOT / public_result["gate_opened_seal_path"]
    )
    assert public_result["gate_result_seal_sha256"] == _sha256(
        REPO_ROOT / public_result["gate_result_seal_path"]
    )
    assert public_result["status"] == "failed_public_gate"
    assert public_result["evaluation_count"] == 1
    assert public_result["public_archive_read_count"] == 1
    assert public_result["exact_scene_count"] == 222
    assert public_result["false_positives"] == 2
    assert public_result["false_negatives"] == 0
    assert public_result["duplicate_region_count"] == 0
    assert public_result["prohibited_structure_hits"] == 2
    assert public_result["case_level_failure_analysis_performed"] is False
    assert public_result["next_revision_may_reuse_public_bytes"] is False
    assert public_result["public_failure_tuning_authorized"] is False
    assert public_result["marker_creation_authorized"] is False
    assert public_result["private_validation_authorized"] is False
    assert public_result["production_approval"] is False
    assert public_result["release_eligible"] is False
    for prohibited in (
        "cases", "predictions", "truths", "fixture_bytes",
        "case_ids", "scene_ids", "proposal_relation_scene_shapes",
    ):
        assert prohibited not in public_result
        assert prohibited not in public_result["metrics"]


def test_public_metric_gate_requires_all_224_scenes_and_direct_bytes() -> None:
    roles = {
        "YTick": 1.0,
        "XTick": 1.0,
        "AxisTitle": 1.0,
        "PhaseHeading": 1.0,
        "LegendText": 1.0,
        "Participant": 1.0,
        "Annotation": 1.0,
        "Other": 1.0,
    }
    metrics: dict[str, object] = {
        "scene_count": 224,
        "truth_region_count": 1792,
        "exact_scene_count": 224,
        "true_positives": 1792,
        "false_positives": 0,
        "false_negatives": 0,
        "duplicate_region_count": 0,
        "prohibited_structure_hits": 0,
        "recognition_exact": 0.90,
        "character_error_rate": 0.05,
        "role_accuracy": 0.90,
        "per_role_accuracy": roles,
        "direct_stored_fixture_byte_execution": True,
    }
    assert _gate_metrics_pass(metrics)
    assert not _gate_metrics_pass({**metrics, "scene_count": 223})
    assert not _gate_metrics_pass({
        **metrics, "direct_stored_fixture_byte_execution": False,
    })


def test_public_runner_seals_before_exactly_one_archive_read() -> None:
    source = inspect.getsource(evaluate_public)
    assert source.count("archive_path.read_bytes()") == 1
    assert source.index("acquire_gate_seal(") < source.index("archive_path.read_bytes()")
    assert "sha256(archive_payload)" in source
    assert "load_archive(BytesIO(archive_payload))" in source
    assert 'runtime_evidence.pop("proposal_relation_scene_shapes", None)' in source


def test_split_freeze_binds_complete_runner_and_aggregate_trigger() -> None:
    expected = {
        ROOT / "PROTOCOL.json",
        ROOT / "dataset.py",
        ROOT / "model.py",
        ROOT / "pipeline.py",
        ROOT / "prepare_split.py",
        ROOT / "protocol.py",
        ROOT / "train_p1.py",
        REPO_ROOT / TRIGGER_RESULT_PATH,
    }
    resolved = {REPO_ROOT / path for path in SOURCE_PATHS}
    assert expected <= resolved
    assert not any("artifacts/public-gate" in path.as_posix() for path in SOURCE_PATHS)


def test_readme_keeps_all_later_gates_unauthorized() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "No V28 case identity" in text
    assert "bytes cannot be used for V29" in text
    assert "zero\nsource-byte overlap" in text
    assert "P1 is checksum-bound" in text
    assert "P1 is selected and consumed" in text
    assert "The run directly executed all 224" in text
    assert "No case identifiers, truth rows, predictions" in text
    assert "V29 is closed" in text
    assert "production\napproval, and release remain unauthorized" in text


def test_python_sources_have_project_spdx_header() -> None:
    for path in ROOT.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# SPDX-License-Identifier: Apache-2.0",
            "# Copyright 2026 Sungwoo Kang",
        ]
