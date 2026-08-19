# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest
import numpy as np
import torch

from ml.ocr.robust_quorum_recall_v31 import dataset as v31_dataset
from ml.ocr.robust_quorum_recall_v31 import prepare_split
from ml.ocr.robust_quorum_recall_v31.dataset import (
    proposal_summary,
    render_scene,
    save_archive,
    split_fingerprint,
)
from ml.ocr.robust_quorum_recall_v31.model import RobustQuorumRecallProposalNet
from ml.ocr.robust_quorum_recall_v31.prepare_split import ARCHIVE_PATHS, SOURCE_PATHS
from ml.ocr.robust_quorum_recall_v31.protocol import (
    REVISION,
    TRIGGER_RESULT_PATH,
    TRIGGER_RESULT_SHA256,
    protocol_configuration,
    split_registration,
)
from ml.ocr.robust_quorum_recall_v31.train_p1 import _trigger_is_terminal
from ml.ocr.robust_quorum_recall_v31 import train_p2, train_p3


REPO_ROOT = Path(__file__).resolve().parents[4]
ROOT = REPO_ROOT / "ml/ocr/robust_quorum_recall_v31"
LEDGER = REPO_ROOT / "ml/markers/training-budgets/production-repair-v1.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _entry() -> dict[str, object]:
    ledger = _read_json(LEDGER)
    return next(
        item for item in ledger["revisions"] if item["revision"] == REVISION
    )


def test_protocol_uses_only_terminal_aggregate_v30_evidence() -> None:
    tracked = _read_json(ROOT / "PROTOCOL.json")
    generated = protocol_configuration()
    assert tracked["schema"] == generated["schema"]
    assert tracked["revision"] == generated["revision"] == REVISION
    assert tracked["trigger_result_path"] == TRIGGER_RESULT_PATH
    assert tracked["trigger_result_sha256"] == TRIGGER_RESULT_SHA256
    assert _sha256(REPO_ROOT / TRIGGER_RESULT_PATH) == TRIGGER_RESULT_SHA256
    basis = tracked["aggregate_design_basis"]
    assert basis["scene_count"] == 256
    assert basis["exact_scene_count"] == 255
    assert basis["truth_regions"] == 2048
    assert basis["true_positives"] == 2047
    assert basis["false_regions"] == 0
    assert basis["missed_regions"] == 1
    assert basis["duplicate_regions"] == 0
    assert basis["prohibited_structure_hits"] == 0
    assert basis["case_level_evidence_used"] is False
    assert basis["fixture_bytes_truth_scene_or_case_identity_reused"] is False


def test_exact_v30_aggregate_trigger_is_terminal_without_case_material() -> None:
    trigger = _read_json(REPO_ROOT / TRIGGER_RESULT_PATH)
    assert _trigger_is_terminal(trigger)
    for prohibited in ("cases", "predictions", "truths", "case_ids", "scene_ids"):
        assert prohibited not in trigger
        assert prohibited not in trigger["metrics"]


def test_split_registrations_are_fresh_and_pairwise_disjoint() -> None:
    registrations = [
        split_registration("train"),
        split_registration("validation"),
        split_registration("sealed_public"),
    ]
    assert [item.scene_count for item in registrations] == [384, 192, 256]
    assert len({item.seed_offset for item in registrations}) == 3
    renderer_sets = [set(item.renderer_families) for item in registrations]
    degradation_sets = [set(item.degradation_families) for item in registrations]
    for left in range(3):
        for right in range(left + 1, 3):
            assert renderer_sets[left].isdisjoint(renderer_sets[right])
            assert degradation_sets[left].isdisjoint(degradation_sets[right])
    assert all("v31" in value for item in registrations for value in (
        *item.renderer_families, *item.degradation_families,
    ))


def test_visible_renderer_is_deterministic_and_proposal_complete() -> None:
    scenes = (
        render_scene("train", 0),
        render_scene("validation", 0),
    )
    repeated = (
        render_scene("train", 0),
        render_scene("validation", 0),
    )
    assert split_fingerprint(scenes[:1]) == split_fingerprint(repeated[:1])
    assert split_fingerprint(scenes[1:]) == split_fingerprint(repeated[1:])
    for scene in scenes:
        summary = proposal_summary((scene,))
        assert summary["exactly_one_production_proposal_per_truth"] is True
        assert summary["positive_proposal_count"] == 8
        assert "v31" in scene.scene_id


def test_renderer_retries_after_post_degradation_proposal_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = render_scene("train", 0)
    source_indices: list[int] = []
    summary_calls = 0

    def fake_render(split: str, index: int):
        source_indices.append(index)
        return source

    def fake_summary(scenes):
        nonlocal summary_calls
        summary_calls += 1
        if summary_calls == 1:
            raise RuntimeError("generic incomplete proposal coverage")
        return {"exactly_one_production_proposal_per_truth": True}

    monkeypatch.setattr(v31_dataset.v30, "render_scene", fake_render)
    monkeypatch.setattr(v31_dataset.v21, "proposal_summary", fake_summary)
    candidate = v31_dataset.render_scene("train", 0)
    assert candidate.scene_id == "robust-quorum-recall-v31-train-00000"
    assert source_indices == [0, 149]


def test_archive_validation_precedes_file_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = render_scene("train", 0)
    archive = tmp_path / "invalid.zip"
    monkeypatch.setattr(
        v31_dataset,
        "proposal_summary",
        lambda _scenes: (_ for _ in ()).throw(RuntimeError("invalid proposals")),
    )
    with pytest.raises(RuntimeError, match="invalid proposals"):
        save_archive((scene,), archive)
    assert not archive.exists()


def test_freeze_cleans_temporary_archives_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene = render_scene("train", 0)
    archives = {
        "train": Path("train.zip"),
        "validation": Path("selection.zip"),
        "sealed_public": Path("public.zip"),
    }
    writes = 0

    def fake_save(_scenes, path: Path):
        nonlocal writes
        writes += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"temporary")
        if writes == 3:
            raise RuntimeError("generic freeze failure")
        return {
            "archive_path": path.as_posix(),
            "archive_sha256": "0" * 64,
            "manifest_sha256": "1" * 64,
            "split_fingerprint": "2" * 64,
            "proposal_summary": {},
        }

    monkeypatch.setattr(prepare_split, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(prepare_split, "SEAL_PATH", Path("SPLIT_SEAL.json"))
    monkeypatch.setattr(prepare_split, "ARCHIVE_PATHS", archives)
    monkeypatch.setattr(prepare_split, "_require_sources_at_head", lambda: None)
    monkeypatch.setattr(prepare_split, "build_split", lambda _split: (scene,))
    monkeypatch.setattr(prepare_split, "save_archive", fake_save)
    with pytest.raises(RuntimeError, match="generic freeze failure"):
        prepare_split.freeze()
    assert not (tmp_path / "SPLIT_SEAL.json").exists()
    for path in archives.values():
        assert not (tmp_path / path).exists()
        assert not (tmp_path / f"{path.as_posix()}.freeze.tmp").exists()


def test_quorum_uses_median_margin_and_tolerates_one_route() -> None:
    attention = torch.tensor([[[4.0, -4.0]]])
    summary = torch.tensor([[[-2.0, 2.0]]])
    local = torch.tensor([[[-3.0, 3.0]]])
    logits = RobustQuorumRecallProposalNet.unanimous_logits(
        attention, summary, local,
    )
    assert logits[0, 0, 1] > logits[0, 0, 0]
    second_negative = torch.tensor([[[3.0, -3.0]]])
    rejected = RobustQuorumRecallProposalNet.unanimous_logits(
        attention, second_negative, local,
    )
    assert rejected[0, 0, 1] < rejected[0, 0, 0]


def test_ledger_consumes_all_v31_candidates_and_keeps_public_closed() -> None:
    entry = _entry()
    assert entry["status"] == "exhausted"
    assert entry["trigger_result_sha256"] == TRIGGER_RESULT_SHA256
    assert entry["trigger_case_detail_or_pixels_used"] is False
    assert entry["prior_fixture_bytes_reused"] is False
    assert entry["prior_checkpoint_reused"] is True
    assert entry["split_materialized"] is True
    assert entry["split_seal_sha256"] == _sha256(ROOT / "SPLIT_SEAL.json")
    assert entry["candidate_config_path"] == (
        "ml/ocr/robust_quorum_recall_v31/training/p3.json"
    )
    assert entry["candidate_config_sha256"] == _sha256(ROOT / "training/p3.json")
    assert entry["split_source_commit"] == _read_json(ROOT / "SPLIT_SEAL.json")[
        "source_commit"
    ]
    assert entry["cross_split_source_overlap_counts"] == {
        "train_validation": 0,
        "train_sealed_public": 0,
        "validation_sealed_public": 0,
    }
    assert entry["preregistered_candidate_ids"] == ["P2", "P3"]
    assert entry["consumed_candidate_ids"] == ["P1", "P2", "P3"]
    assert entry["remaining_unregistered_candidate_ids"] == []
    assert entry["selection_evaluations"] == 2
    assert entry["p1_status"] == "failed_runner_consumed"
    assert entry["p1_optimizer_steps"] == 0
    assert entry["p1_selection_archive_read_count"] == 1
    assert entry["p1_case_detail_or_pixels_inspected"] is False
    assert entry["p2_status"] == "failed_selection"
    assert entry["p2_optimizer_steps"] == 0
    assert entry["p2_selection_archive_read_count"] == 1
    assert entry["p2_passing_threshold_window"] == []
    assert entry["p2_selected_threshold_metrics"]["exact_scene_count"] == 192
    assert entry["p2_selected_threshold_metrics"]["false_positives"] == 0
    assert entry["p2_lower_threshold_aggregate"]["false_positives_each"] == 1
    assert entry["p2_case_detail_or_pixels_inspected"] is False
    assert entry["p2_result_sha256"] == _sha256(ROOT / "P2_RESULT.json")
    assert entry["p2_candidate_config_sha256"] == _sha256(ROOT / "training/p2.json")
    assert entry["p3_expected_optimizer_steps"] == 1536
    p3_result = _read_json(ROOT / "P3_RESULT.json")
    assert entry["p3_expected_runner_source_bundle_sha256"] == p3_result[
        "invocation_runner_source_bundle_sha256"
    ]
    assert entry["p3_status"] == "failed_authorization_contract_consumed"
    assert entry["p3_optimizer_steps"] == 0
    assert entry["p3_selection_archive_read_count"] == 0
    assert entry["p3_result_sha256"] == _sha256(ROOT / "P3_RESULT.json")
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert entry["marker_creation_evaluated"] is False
    assert entry["private_validation"] is False
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False


def test_frozen_split_and_consumed_candidates_are_bound_while_public_remains_closed() -> None:
    seal_path = ROOT / "SPLIT_SEAL.json"
    config_path = ROOT / "training/p1.json"
    seal = _read_json(seal_path)
    config = _read_json(config_path)
    assert _sha256(seal_path) == config["split_seal_sha256"]
    assert seal["source_bundle_sha256"] == config[
        "expected_runner_source_bundle_sha256"
    ]
    assert seal["cross_split_source_overlap_counts"] == {
        "train_validation": 0,
        "train_sealed_public": 0,
        "validation_sealed_public": 0,
    }
    assert seal["candidate_execution_authorized"] is False
    assert config["candidate_execution_authorized"] is True
    assert config["public_execution_authorized"] is False
    for split, path in ARCHIVE_PATHS.items():
        assert _sha256(REPO_ROOT / path) == seal["splits"][split]["archive_sha256"]
        assert len(seal["splits"][split]["source_sha256_inventory"]) == seal[
            "splits"
        ][split]["proposal_summary"]["scene_count"]
    result = _read_json(ROOT / "P1_RESULT.json")
    assert result["status"] == "failed_runner_consumed"
    assert result["selection_archive_read_count"] == 1
    assert result["public_gate_archive_opened"] is False
    assert result["case_detail_or_pixels_inspected"] is False
    assert _sha256(ROOT / "artifacts/P1-run/candidate-report.json") == result[
        "candidate_report_sha256"
    ]
    p2_result = _read_json(ROOT / "P2_RESULT.json")
    assert p2_result["status"] == "failed_selection"
    assert p2_result["selection_archive_read_count"] == 1
    assert p2_result["passing_threshold_window"] == []
    assert p2_result["selected_threshold_metrics"]["exact_scene_count"] == 192
    assert p2_result["selected_threshold_metrics"]["false_positives"] == 0
    assert p2_result["threshold_comparisons"][0]["false_positives"] == 1
    assert p2_result["case_detail_or_pixels_inspected"] is False
    assert p2_result["public_gate_archive_opened"] is False
    local_report = ROOT / p2_result["candidate_report_path"].split(
        "robust_quorum_recall_v31/", maxsplit=1,
    )[1]
    if local_report.exists():
        assert _sha256(local_report) == p2_result["candidate_report_sha256"]


def test_consumed_p2_cannot_pass_preflight_again() -> None:
    with pytest.raises(RuntimeError):
        train_p2.preflight(require_authorized=False)
    with pytest.raises(RuntimeError):
        train_p2.preflight(require_authorized=True)


def test_p3_is_checksum_bound_consumed_and_cannot_reenter_selection() -> None:
    config = _read_json(ROOT / "training/p3.json")
    entry = _entry()
    assert config["candidate_id"] == "P3"
    assert config["candidate_execution_authorized"] is True
    assert config["selection_thresholds"] == [0.35, 0.45, 0.55, 0.65, 0.75]
    assert config["minimum_consecutive_passing_thresholds"] == 3
    assert config["validation_or_public_pixels_used_for_training"] is False
    assert config["case_level_predecessor_evidence_used"] is False
    result = _read_json(ROOT / "P3_RESULT.json")
    assert config["expected_runner_source_bundle_sha256"] == result[
        "invocation_runner_source_bundle_sha256"
    ]
    assert result["single_execution_invocation_count"] == 1
    assert result["training_started"] is False
    assert result["selection_archive_opened"] is False
    assert result["rerun_authorized"] is False
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["public_gate_authorized"] is False
    assert not (ROOT / "artifacts/P3-run").exists()


def test_p3_preflight_mirrors_single_candidate_acquisition_contract() -> None:
    historical_entry = {
        "execution_authorized": True,
        "authorized_candidate_id": "P3",
        "preregistered_candidate_ids": ["P2", "P3"],
        "consumed_candidate_ids": ["P1", "P2"],
    }
    assert not train_p3._single_candidate_acquisition_contract_satisfied(
        historical_entry,
    )


def test_p3_loader_replaces_the_complete_predecessor_state() -> None:
    predecessor = RobustQuorumRecallProposalNet(seed=17)
    candidate = train_p3.MarginFineTunedRobustQuorumRecallProposalNet(seed=29)
    expected = {name: value.detach().clone() for name, value in predecessor.state_dict().items()}
    candidate.load_role_parent_state_dict(expected)
    assert all(
        torch.equal(value, expected[name])
        for name, value in candidate.state_dict().items()
    )


def test_p2_ort_adapter_is_callable_contiguous_and_float32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class Input:
        name = "input"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, outputs, feeds):
            observed["outputs"] = outputs
            observed["values"] = feeds["input"]
            return [np.asarray(feeds["input"], dtype=np.float64) + 1.0]

    monkeypatch.setattr(train_p2, "_RAW_CPU_SESSION", lambda _path: Session())
    runner = train_p2._callable_cpu_session(Path("fixed.onnx"))
    source = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
    result = runner(source)
    values = observed["values"]
    assert isinstance(values, np.ndarray)
    assert values.flags.c_contiguous
    assert values.dtype == np.float32
    assert result.dtype == np.float32
    assert np.array_equal(result, values + 1.0)


def test_source_inventory_binds_v31_runner_and_v30_aggregate_trigger() -> None:
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
    assert ROOT / "train_p2.py" in {
        REPO_ROOT / path for path in train_p2.RUNNER_SOURCE_PATHS
    }
    assert ROOT / "P1_RESULT.json" in {
        REPO_ROOT / path for path in train_p2.RUNNER_SOURCE_PATHS
    }


def test_readme_forbids_public_reuse_and_application_synthetic_data() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    assert "No V30 case identity" in text
    assert "V30 public bytes and case identities cannot be reused" in text
    assert "zero optimizer steps" in text
    assert "P1 is consumed" in text
    assert "P2 is consumed and failed visible selection" in text
    assert "P3 is consumed and V31 is exhausted" in text
    assert "the selection archive was not opened" in normalized
    assert "consumed P3 invocation cannot be rerun" in normalized
    assert "Public execution" in normalized and "remain closed" in normalized
    assert "never become application graph data" in normalized


def test_python_sources_have_project_spdx_header() -> None:
    for path in ROOT.rglob("*.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == [
            "# SPDX-License-Identifier: Apache-2.0",
            "# Copyright 2026 Sungwoo Kang",
        ]


@pytest.mark.parametrize("split", ("train", "validation", "sealed_public"))
def test_split_names_reject_predecessor_identity(split: str) -> None:
    registration = split_registration(split)
    assert all("v30" not in value for value in (
        *registration.renderer_families, *registration.degradation_families,
    ))
