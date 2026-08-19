# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
)


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
    }
    assert entry["prior_fixture_bytes_reused"] is False
    assert entry["trigger_case_detail_or_pixels_used"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    if entry["status"] == "preregistered_source_only":
        assert entry["split_materialized"] is False
        assert not (REPO_ROOT / SEAL_PATH).exists()
        assert all(not (REPO_ROOT / path).exists() for path in ARCHIVE_PATHS.values())


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
    assert "No V29 fixture archive" in text
    assert "production approval, and release\nremain unauthorized" in text


def test_python_sources_have_project_spdx_header() -> None:
    for path in ROOT.glob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# SPDX-License-Identifier: Apache-2.0",
            "# Copyright 2026 Sungwoo Kang",
        ]
