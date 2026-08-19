# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import torch

from ml.ocr.unanimous_structure_veto_v30.model import (
    UnanimousStructureVetoProposalNet,
)
from ml.ocr.unanimous_structure_veto_v30.protocol import (
    REVISION,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
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
