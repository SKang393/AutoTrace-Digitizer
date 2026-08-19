# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

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


def test_fresh_split_registrations_are_disjoint_and_not_materialized() -> None:
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
    assert not (ROOT / "SPLIT_SEAL.json").exists()
    for name in ("train", "selection", "public"):
        assert not (
            REPO_ROOT / f"artifacts/production-validation/ocr-v30-{name}.zip"
        ).exists()


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


def test_runner_source_is_complete_but_training_remains_fail_closed() -> None:
    required = {
        "dataset.py", "model.py", "pipeline.py", "prepare_split.py",
        "protocol.py", "train_p1.py",
    }
    assert required <= {path.name for path in SOURCE_PATHS if path.parent == Path(
        "ml/ocr/unanimous_structure_veto_v30"
    )}
    with pytest.raises(RuntimeError, match="OCR V30 P1 config field mismatch"):
        preflight()


def test_ledger_refuses_execution_and_every_later_gate() -> None:
    entry = _entry()
    assert entry["status"] == "preregistered_source_only"
    assert entry["prior_revision"] == "graph-text-dual-route-consensus-proposal-v29"
    assert entry["trigger_result_sha256"] == TRIGGER_RESULT_SHA256
    assert entry["trigger_case_detail_or_pixels_used"] is False
    assert entry["prior_fixture_bytes_reused"] is False
    assert entry["prior_checkpoint_reused"] is False
    assert entry["split_materialized"] is False
    assert entry["preregistered_candidate_ids"] == []
    assert entry["consumed_candidate_ids"] == []
    assert entry["execution_authorized"] is False
    assert entry["public_gate_authorized"] is False
    assert entry["marker_creation_evaluated"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_readme_forbids_v29_bytes_and_application_synthetic_data() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "No V29 case identity" in text
    assert "bytes cannot be used for V30" in text
    assert "No V29 checkpoint is reused" in text
    assert "This checkpoint is source-only" in text
    assert "never become application graph data" in normalized


def test_python_sources_have_project_spdx_header() -> None:
    for path in ROOT.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# SPDX-License-Identifier: Apache-2.0",
            "# Copyright 2026 Sungwoo Kang",
        ]
