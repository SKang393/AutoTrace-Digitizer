# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from ml.markers.gate_seal import source_bundle_sha256

from ml.ocr.unanimous_structure_veto_v30.dataset import (
    proposal_summary,
    render_scene,
    split_fingerprint,
)
from ml.ocr.unanimous_structure_veto_v30.model import (
    UnanimousStructureVetoProposalNet,
)
from ml.ocr.unanimous_structure_veto_v30.prepare_split import SOURCE_PATHS
from ml.ocr.unanimous_structure_veto_v30.protocol import (
    REVISION,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.unanimous_structure_veto_v30.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    PUBLIC_CONFIG_PATH,
    PUBLIC_OUTPUT_PATH,
    _aggregate_metrics,
    _gate_metrics_pass,
    _selected_result_is_terminal,
    _validate_config,
    evaluate_public,
)
from ml.ocr.unanimous_structure_veto_v30.train_p1 import (
    _trigger_is_terminal,
    _unanimous_objective,
    preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/unanimous_structure_veto_v30"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _entry() -> dict[str, object]:
    ledger = _read_json(LEDGER)
    return next(
        item for item in ledger["revisions"]
        if item["revision"] == REVISION
    )


def test_protocol_is_source_only_and_uses_aggregate_v29_evidence() -> None:
    tracked = _read_json(ROOT / "PROTOCOL.json")
    generated = protocol_configuration()
    assert tracked["schema"] == generated["schema"]
    assert tracked["revision"] == generated["revision"] == REVISION
    assert tracked["trigger_result_path"] == TRIGGER_RESULT_PATH
    assert tracked["trigger_result_sha256"] == TRIGGER_RESULT_SHA256
    assert _sha256(REPO_ROOT / TRIGGER_RESULT_PATH) == TRIGGER_RESULT_SHA256
    basis = tracked["aggregate_design_basis"]
    assert basis["scene_count"] == 224
    assert basis["exact_scene_count"] == 222
    assert basis["true_positives"] == basis["truth_regions"] == 1792
    assert basis["false_regions"] == 2
    assert basis["missed_regions"] == 0
    assert basis["duplicate_regions"] == 0
    assert basis["prohibited_structure_hits"] == 2
    assert basis["case_level_evidence_used"] is False
    assert basis["fixture_bytes_truth_scene_or_case_identity_reused"] is False
    assert tracked["fixture_identity_frozen"] is False
    assert tracked["training_authorized"] is False
    assert tracked["public_execution_authorized"] is False
    assert tracked["private_validation_authorized"] is False
    assert tracked["production_approval"] is False
    assert tracked["release_eligible"] is False


def test_fresh_split_registrations_are_disjoint_and_checksum_sealed() -> None:
    registrations = [
        split_registration("train"),
        split_registration("validation"),
        split_registration("sealed_public"),
    ]
    assert [item.scene_count for item in registrations] == [384, 192, 256]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_ids = [value for item in registrations for value in item.renderer_families]
    degradation_ids = [
        value for item in registrations for value in item.degradation_families
    ]
    assert len(renderer_ids) == len(set(renderer_ids))
    assert len(degradation_ids) == len(set(degradation_ids))
    seal_path = ROOT / "SPLIT_SEAL.json"
    seal = _read_json(seal_path)
    config = _read_json(ROOT / "training/p1.json")
    assert _sha256(seal_path) == config["split_seal_sha256"]
    assert seal["source_commit"] == "380d4ece1b48623e60b4d82720d8b421d97349f3"
    assert seal["source_bundle_sha256"] == config["split_source_bundle_sha256"]
    assert set(seal["cross_split_source_overlap_counts"].values()) == {0}
    assert seal["optimizer_steps_at_freeze"] == 0
    assert seal["selection_evaluations"] == 0
    assert seal["public_evaluations"] == 0
    assert seal["training_authorized"] is False
    archive_bindings = {
        "train": "train",
        "validation": "selection",
        "sealed_public": "public",
    }
    for split, name in archive_bindings.items():
        path = REPO_ROOT / f"artifacts/production-validation/ocr-v30-{name}.zip"
        assert _sha256(path) == seal["splits"][split]["archive_sha256"]
        assert seal["splits"][split]["proposal_summary"][
            "exactly_one_production_proposal_per_truth"
        ] is True


def test_unanimous_margin_allows_any_route_to_veto() -> None:
    attention = torch.tensor([[[0.0, 4.0], [0.0, 4.0]]])
    summary = torch.tensor([[[0.0, 3.0], [0.0, 3.0]]])
    local = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    logits = UnanimousStructureVetoProposalNet.unanimous_logits(
        attention, summary, local,
    )
    margins = logits[:, :, 1] - logits[:, :, 0]
    assert torch.equal(margins, torch.tensor([[2.0, -2.0]]))


def test_random_weight_model_contract_and_deterministic_roles() -> None:
    torch.manual_seed(3001)
    model = UnanimousStructureVetoProposalNet().eval()
    evidence = torch.randn(1, 5, 31)
    crops = torch.randn(1, 5, 2, 32, 128)
    relations = torch.randn(1, 5, 5, 19)
    with torch.no_grad():
        output = model(evidence, crops, relations)
        consensus, attention, summary, local = model.proposal_routes(
            evidence, crops, relations,
        )
    assert output.shape == (1, 5, 10)
    assert consensus.shape == attention.shape == summary.shape == local.shape == (
        1, 5, 2,
    )
    assert torch.equal(output[:, :, :2], consensus)
    assert torch.equal(output[:, :, 2:], model.role_logits(evidence))


def test_model_is_permutation_equivariant_over_proposals() -> None:
    torch.manual_seed(3002)
    model = UnanimousStructureVetoProposalNet().eval()
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
    torch.manual_seed(3003)
    model = UnanimousStructureVetoProposalNet().eval()
    evidence = torch.randn(1, 7, 31)
    crops = torch.randn(1, 7, 2, 32, 128)
    relations = torch.randn(1, 7, 7, 19)
    path = tmp_path / "v30-contract.onnx"
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


def test_fresh_renderer_is_deterministic_and_complete() -> None:
    scenes = tuple(render_scene(split, 0) for split in (
        "train", "validation", "sealed_public",
    ))
    repeated = render_scene("train", 0)
    assert repeated.scene_id == scenes[0].scene_id
    assert torch.equal(
        torch.from_numpy(repeated.raster), torch.from_numpy(scenes[0].raster),
    )
    assert len({scene.scene_id for scene in scenes}) == 3
    assert len({split_fingerprint((scene,)) for scene in scenes}) == 3
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert summary["positive_proposal_count"] == 8
        assert "v29" not in scene.scene_id


def test_unanimous_objective_trains_every_route_and_rewards_diversity() -> None:
    consensus = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], requires_grad=True)
    attention = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], requires_grad=True)
    summary = torch.tensor([[0.75, -0.75], [-0.75, 0.75]], requires_grad=True)
    local = torch.tensor([[1.25, -1.25], [-1.25, 1.25]], requires_grad=True)
    targets = torch.tensor([0, 1])
    weights = torch.ones(2)
    config = _read_json(ROOT / "training/p1.json")
    loss, parts = _unanimous_objective(
        consensus, attention, summary, local, targets, weights, config,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert parts["pairwise_route_diversity_reward"] > 0
    assert all(route.grad is not None for route in (
        consensus, attention, summary, local,
    ))


def test_exact_v29_aggregate_trigger_is_terminal_without_case_material() -> None:
    trigger = _read_json(REPO_ROOT / TRIGGER_RESULT_PATH)
    assert _trigger_is_terminal(trigger)
    for prohibited in ("cases", "predictions", "truths"):
        assert prohibited not in trigger


def test_consumed_runner_preflight_refuses_a_second_execution() -> None:
    required = {
        "dataset.py", "model.py", "pipeline.py", "prepare_split.py",
        "protocol.py", "train_p1.py",
    }
    assert required <= {path.name for path in SOURCE_PATHS if path.parent == Path(
        "ml/ocr/unanimous_structure_veto_v30"
    )}
    with pytest.raises(RuntimeError, match="P1 output already exists"):
        preflight()


def test_ledger_records_selected_p1_and_keeps_later_gates_closed() -> None:
    entry = _entry()
    result_path = ROOT / "P1_RESULT.json"
    result = _read_json(result_path)
    assert entry["status"] == "candidate_1_selected_public_gate_preregistered"
    assert entry["prior_revision"] == "graph-text-dual-route-consensus-proposal-v29"
    assert entry["trigger_result_sha256"] == TRIGGER_RESULT_SHA256
    assert entry["trigger_case_detail_or_pixels_used"] is False
    assert entry["prior_fixture_bytes_reused"] is False
    assert entry["prior_checkpoint_reused"] is False
    assert entry["split_materialized"] is True
    assert entry["split_source_commit"] == "380d4ece1b48623e60b4d82720d8b421d97349f3"
    assert entry["split_seal_path"] == (
        "ml/ocr/unanimous_structure_veto_v30/SPLIT_SEAL.json"
    )
    assert entry["split_seal_sha256"] == _sha256(ROOT / "SPLIT_SEAL.json")
    assert entry["candidate_config_sha256"]["P1"] == _sha256(
        ROOT / "training/p1.json"
    )
    assert _read_json(ROOT / "training/p1.json")["training_authorized"] is True
    assert entry["preregistered_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == ["P1"]
    assert entry["selection_evaluations"] == 1
    assert entry["candidate_1_result_path"] == (
        "ml/ocr/unanimous_structure_veto_v30/P1_RESULT.json"
    )
    assert entry["candidate_1_result_sha256"] == _sha256(result_path)
    assert entry["candidate_1_report_sha256"] == result["report_sha256"]
    assert entry["candidate_1_checkpoint_sha256"] == result["checkpoint_sha256"]
    assert entry["candidate_1_onnx_sha256"] == result["onnx_sha256"]
    assert entry["candidate_1_selected_threshold"] == 0.55
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["marker_creation_evaluated"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_p1_result_is_aggregate_only_and_passes_fixed_selection_gates() -> None:
    result = _read_json(ROOT / "P1_RESULT.json")
    assert result["status"] == "selected"
    assert result["candidate_consumed"] is True
    assert result["optimizer_steps"] == 1536
    assert result["selection_evaluations"] == 1
    assert result["selection_gate_passed"] is True
    assert result["selected_threshold"] == 0.55
    assert result["passing_threshold_window"] == [0.35, 0.45, 0.55, 0.65, 0.75]
    metrics = result["selection_metrics"]
    assert metrics["scene_count"] == metrics["exact_scene_count"] == 192
    assert metrics["truth_region_count"] == metrics["true_positives"] == 1536
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["duplicate_region_count"] == 0
    assert metrics["prohibited_structure_hits"] == 0
    assert metrics["recognition_exact"] >= 0.90
    assert metrics["character_error_rate"] <= 0.05
    assert metrics["role_accuracy"] == 1.0
    assert set(metrics["per_role_accuracy"].values()) == {1.0}
    assert metrics["direct_stored_fixture_byte_execution"] is True
    assert result["provider"] == "CPUExecutionProvider"
    assert result["onnx_parity_maximum_absolute_error"] <= 1e-5
    assert result["onnx_parity_passed"] is True
    assert result["case_level_details_emitted"] is False
    for prohibited in ("cases", "predictions", "truths", "case_ids"):
        assert prohibited not in result
    assert result["public_gate_evaluations"] == 0
    assert result["public_gate_archive_opened"] is False
    assert result["public_gate_authorized"] is False
    assert result["production_approval"] is False
    assert result["release_eligible"] is False


def test_public_gate_is_preregistered_but_not_authorized() -> None:
    config_path = REPO_ROOT / PUBLIC_CONFIG_PATH
    config = _read_json(config_path)
    _validate_config(config, require_authorized=False)
    with pytest.raises(RuntimeError, match="not separately authorized"):
        _validate_config(config, require_authorized=True)
    assert config["runner_source_commit"] is None
    assert config["public_execution_authorized"] is False
    assert config["expected_evaluator_source_bundle_sha256"] == (
        source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS)
    )
    assert not (REPO_ROOT / PUBLIC_OUTPUT_PATH).exists()
    entry = _entry()
    assert entry["status"] == "candidate_1_selected_public_gate_preregistered"
    assert entry["public_gate_config_path"] == PUBLIC_CONFIG_PATH.as_posix()
    assert entry["public_gate_config_sha256"] == _sha256(config_path)
    assert entry["public_gate_runner_source_commit"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["public_gate_archive_opened"] is False


def test_selected_p1_is_terminal_for_public_gate_without_case_material() -> None:
    result = _read_json(ROOT / "P1_RESULT.json")
    assert _selected_result_is_terminal(result)
    for prohibited in (
        "cases", "predictions", "truths", "fixture_bytes", "scene_ids",
        "case_ids", "proposal_relation_scene_shapes",
    ):
        assert prohibited not in result


def test_public_metric_gate_requires_all_256_scenes_and_direct_bytes() -> None:
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
        "scene_count": 256,
        "truth_region_count": 2048,
        "exact_scene_count": 256,
        "true_positives": 2048,
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
    assert not _gate_metrics_pass({**metrics, "scene_count": 255})
    assert not _gate_metrics_pass({
        **metrics, "direct_stored_fixture_byte_execution": False,
    })


def test_public_report_whitelists_aggregate_metrics_only() -> None:
    selection = dict(_read_json(ROOT / "P1_RESULT.json")["selection_metrics"])
    selection["public_archive_read_count"] = 1
    selection["cases"] = ["must-not-escape"]
    selection["proposal_relation_scene_shapes"] = [[1, 1, 19]]
    aggregate = _aggregate_metrics(selection)
    assert "cases" not in aggregate
    assert "proposal_relation_scene_shapes" not in aggregate
    assert aggregate["scene_count"] == 192
    assert aggregate["public_archive_read_count"] == 1


def test_public_runner_seals_before_one_read_and_drops_scene_shapes() -> None:
    source = inspect.getsource(evaluate_public)
    assert source.count(".read_bytes()") == 1
    assert source.index("acquire_gate_seal(") < source.index(".read_bytes()")
    assert "sha256(archive_payload)" in source
    assert "load_archive(BytesIO(archive_payload))" in source
    assert 'runtime_evidence.pop("proposal_relation_scene_shapes", None)' in source
    assert "_aggregate_metrics(selected_metrics)" in source
    assert "_aggregate_comparison(item)" in source


def test_readme_forbids_v29_bytes_and_application_synthetic_data() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "No V29 case identity" in text
    assert "bytes cannot be used for V30" in text
    assert "No V29 checkpoint is reused" in text
    assert "P1 consumed its single authorized CPU training run" in text
    assert "sealed public archive remains unopened" in text
    assert "configuration remains unauthorized" in text
    assert "writes only whitelisted aggregate metrics" in text
    assert "never become application graph data" in normalized


def test_python_sources_have_project_spdx_header() -> None:
    for path in ROOT.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# SPDX-License-Identifier: Apache-2.0",
            "# Copyright 2026 Sungwoo Kang",
        ]
