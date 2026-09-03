# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
import hashlib
from pathlib import Path

from ml.markers.center.mask_preserving_v24.train_p1 import RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import source_bundle_sha256

ROOT = Path(__file__).resolve().parents[5]

def test_training_contract_is_fixed_and_candidate_not_run():
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    assert config["seed"] == 20260903
    assert config["confidence_threshold"] == 0.25
    assert config["selection_thresholds"] == [0.40, 0.55, 0.70]
    assert config["optimizer_steps_expected"] == 10080
    assert config["optimizer_steps_maximum"] == 10080
    assert config["training_example_count_expected"] == 35838
    assert config["retry_count"] == 3
    assert config["negative_sampler"]["total_expected"] == 32580
    assert config["negative_sampler"]["selected_index_sha256"] == "943199e6b93c7d02a3eb98506aad03ba69ee4d45024cc707c8f9efdbe71c5649"
    assert sum(config["negative_sampler"]["quotas"].values()) == 32580
    assert config["sealed_runs"] == 0 and config["private_data"] is False
    assert config["real_dev_reads"] == 0 and config["real_sealed_reads"] == 0

def test_runner_source_bundle_is_relative_and_present():
    assert all(not path.is_absolute() for path in RUNNER_SOURCE_PATHS)
    assert all((ROOT / path).is_file() for path in RUNNER_SOURCE_PATHS)
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == config["revision"])
    assert entry["p1_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    if entry["execution_authorized"] or entry["status"] == "candidate_1_preregistered":
        assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]

def test_current_evidence_bindings_and_authorization_match_files():
    config_path = ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json"
    config = json.loads(config_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for path_key, hash_key in (
        ("morphology_diagnosis_path", "morphology_diagnosis_sha256"),
        ("morphology_gap_path", "morphology_gap_sha256"),
        ("generator_audit_path", "generator_audit_sha256"),
        ("negative_audit_path", "negative_audit_sha256"),
    ):
        assert digest(ROOT / config[path_key]) == config[hash_key]
    audit = json.loads((ROOT / config["generator_audit_path"]).read_text())
    assert audit["splits"]["train"]["aggregate_sha256"] == config["train_split_sha256"]
    assert audit["splits"]["dev"]["aggregate_sha256"] == config["dev_split_sha256"]
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == config["revision"])
    assert digest(config_path) == entry["candidate_config_sha256"]["P1"]
    assert entry["expected_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    assert digest(ROOT / "ml/markers/center/real_range_generator_v1/generator.py") == entry["negative_generator_sha256"]
    assert digest(ROOT / "ml/markers/center/real_range_generator_v1/negative_proposal_audit.py") == entry["synthetic_negative_audit_source_sha256"]
    assert digest(ROOT / "ml/markers/center/real_range_generator_v1/AUDIT.json") == entry["negative_generator_audit_sha256"]
    assert digest(ROOT / "ml/markers/center/real_range_generator_v1/NEGATIVE_PROPOSAL_AUDIT.json") == entry["synthetic_negative_audit_sha256"]
    assert entry["synthetic_negative_proposal_count"] == 237578
    assert entry["morphology_diagnosis_sha256"] == config["morphology_diagnosis_sha256"]
    assert entry["morphology_gap_sha256"] == config["morphology_gap_sha256"]
    assert entry["status"] == "dev_passed_retry3_unconsumed"
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["real_dev_authorized"] is False
    assert entry["real_sealed_authorized"] is False
    assert entry["real_sealed_reads"] == 0
    assert entry["sealed_runs"] == 0
    assert entry["consumed_candidate_ids"] == []
    assert entry["dev_passed_candidate_ids"] == ["P1"]
    assert entry["candidate_consumed"] is False
    assert entry["p1_result_path"].endswith("P1_RETRY3_RESULT.json")
    assert digest(ROOT / entry["p1_result_path"]) == entry["p1_result_sha256"]
    assert entry["p1_result_sha256"] == "edf2ba744146fdcb6407b68d5765784f733b2f2e8739ecceedfd651edb372711"
    assert entry["p1_candidate_report_sha256"] == entry["p1_result_sha256"]
    assert entry["p1_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    assert digest(ROOT / entry["retry3_morphology_diagnosis_path"]) == entry["retry3_morphology_diagnosis_sha256"]
    assert entry["retry3_morphology_accepted_generic_false_positives"] == 16
    assert entry["retry3_morphology_scene_count"] == 167
    assert entry["retry3_morphology_optimizer_steps"] == 0
    assert entry["retry3_morphology_private_data"] is False
    assert entry["retry3_morphology_real_dev_reads"] == 0
    assert entry["retry3_morphology_real_sealed_reads"] == 0
    assert entry["retry3_morphology_sealed_runs"] == 0
    assert entry["retry3_morphology_threshold_change_proposed"] is False
    assert entry["p1_checkpoint_sha256"] == "70b9947bdaa78d5465f7cd2026a4bc00fd3805507551c002daf763e5dbc0b318"
    assert entry["p1_onnx_sha256"] == "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"
    assert entry["p1_true_positives"] == 1977
    assert entry["p1_false_positives"] == 16
    assert entry["p1_false_negatives"] == 27
    assert entry["p1_precision"] == 0.9919719016557953
    assert entry["p1_recall"] == 0.9865269461077845
    assert entry["p1_f1"] == 0.9892419314485864
    assert entry["p1_onnx_parity_maximum_absolute_error"] == 1.9073486328125e-06
    assert entry["p1_opened_seal_sha256"] == "33773333d6f75814c4f0801d4c86438990a8478eddb021b7912bc4ea8bb6ebee"
    assert entry["p1_result_seal_sha256"] == "d8e830b2e384a67a1da3911ba774c707a5cac66ded3bfbabf2870e54c2043ee5"
    assert entry["p1_dev_gate_passed"] is True
    assert entry["retry2_dev_gate_passed"] is True
    assert entry["spatial_morphology_diagnostic_required"] is False
    assert entry["execution_blocker"]

def test_train_examples_preserve_mask_crossing_positive():
    from ml.markers.center.mask_preserving_v24.train_p1 import _examples
    from ml.markers.center.real_range_generator_v1.generator import build_split
    import torch
    scene = next(
        item for item in build_split("train")
        if any(
            float(item.tensor[1, int(y)-2:int(y)+3, int(x)-2:int(x)+3].max()) >= .35
            for x, y in item.centers
        )
    )
    patches, labels, _, radii, _ = _examples((scene,), 10, torch.Generator().manual_seed(20260904))
    assert patches.shape[1:] == (3, 33, 33)
    assert bool((labels > 0.5).any())
    assert bool((patches[labels > 0.5, 1].sum(dim=(1,2)) > 0).any())
    assert float(radii.max()) <= max(scene.diameters) / 2.0
