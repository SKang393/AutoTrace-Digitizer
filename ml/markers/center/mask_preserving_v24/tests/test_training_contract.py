# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
import hashlib
from pathlib import Path

from ml.markers.center.mask_preserving_v24.train_p1 import RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import source_bundle_sha256
from ml.markers.center.real_range_generator_v1.negative_sampler import TOPOLOGY_RADIUS_PX

ROOT = Path(__file__).resolve().parents[5]

def test_training_contract_is_fixed_and_candidate_not_run():
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    assert config["seed"] == 20260903
    assert config["confidence_threshold"] == 0.25
    assert config["selection_thresholds"] == [0.40, 0.55, 0.70]
    assert config["optimizer_steps_expected"] == 10080
    assert config["optimizer_steps_maximum"] == 10080
    assert config["training_example_count_expected"] == 35838
    assert config["retry_count"] == 5
    assert config["negative_sampler"]["total_expected"] == 32580
    assert config["negative_sampler"]["source_sha256"] == "ca608bba006dd4ff5a0b525829e9a61f9b64673cf82fca06cc087f1e9654a858"
    assert config["negative_sampler"]["selected_index_sha256"] == "a81fb8127cda6819f1d0da318dc34ea2891dec5e3c6c767eb756af68bd2f869f"
    assert config["negative_sampler"]["expected_capacities"] == {"artifact": 13488, "faint_low": 8598, "faint_p05": 5127, "generic": 179238, "hard_existing": 6012, "ocr_heavy": 18748}
    assert config["negative_sampler"]["topology"] == {"radius_px": 16.0, "expected_capacity": {"topology_junction": 8331, "topology_fragment": 8049}, "expected_selected": {"topology_junction": 8331, "topology_fragment": 8049}, "selected_index_sha256": "df24e495a76d485adcef07defd96ee723da3e029ededd5009e9c34e7ac58325d"}
    assert config["negative_sampler"]["topology"]["radius_px"] == TOPOLOGY_RADIUS_PX
    assert sum(config["negative_sampler"]["quotas"].values()) == 32580
    assert config["sealed_runs"] == 0 and config["private_data"] is False
    assert config["real_dev_reads"] == 0 and config["real_sealed_reads"] == 0
    assert config["retry3_morphology_gap_sha256"] == "3b0e9981eb3d21787679f1df1151a3c0bc395ce7966e6c681c6de5755c3fb769"
    assert config["retry4_diagnosis_sha256"] == "a19745f7904c8ec316a78a4e220e3133fc5f77fa80f471ed5337976bdbb6594b"
    assert config["retry4_generic_fp_diagnosis_sha256"] == "24d86878dc335803b2aacd6bab5105496cbb2fb51734b4eb0d8ead4feea5d172"
    assert config["negative_sampler"]["connector"] == {"fractions": [0.3333333333333333, 0.6666666666666666], "max_distance_px": 4.0, "target_count": 3674, "expected_capacity": 3661, "expected_selected": 3661, "selected_index_sha256": "40a14e3becc3f793a590cc2b37fdd92107a97f71f59c6fc1cf9d22d4d124116d"}

def test_runner_source_bundle_is_relative_and_present():
    assert all(not path.is_absolute() for path in RUNNER_SOURCE_PATHS)
    assert all((ROOT / path).is_file() for path in RUNNER_SOURCE_PATHS)
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == config["revision"])
    assert entry["p1_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    assert entry["execution_authorized"] is True
    assert entry["status"] == "candidate_1_preregistered"
    assert source_bundle_sha256(ROOT, RUNNER_SOURCE_PATHS) == config["expected_runner_source_bundle_sha256"]

def test_current_evidence_bindings_and_authorization_match_files():
    config_path = ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json"
    config = json.loads(config_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for path_key, hash_key in (
        ("morphology_diagnosis_path", "morphology_diagnosis_sha256"),
        ("morphology_gap_path", "morphology_gap_sha256"),
        ("retry3_morphology_gap_path", "retry3_morphology_gap_sha256"),
        ("retry4_diagnosis_path", "retry4_diagnosis_sha256"),
        ("retry4_generic_fp_diagnosis_path", "retry4_generic_fp_diagnosis_sha256"),
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
    assert entry["synthetic_negative_proposal_count"] == 231211
    assert entry["morphology_diagnosis_sha256"] == config["morphology_diagnosis_sha256"]
    assert entry["morphology_gap_sha256"] == config["morphology_gap_sha256"]
    assert entry["status"] == "candidate_1_preregistered"
    assert entry["execution_authorized"] is True
    assert entry["authorized_candidate_id"] == "P1"
    assert entry["real_dev_authorized"] is False
    assert entry["real_sealed_authorized"] is False
    assert entry["real_sealed_reads"] == 0
    assert entry["sealed_runs"] == 0
    assert entry["consumed_candidate_ids"] == []
    assert entry["dev_passed_candidate_ids"] == []
    assert entry["candidate_consumed"] is False
    assert entry["p1_result_path"].endswith("P1_RETRY4_RESULT.json")
    assert digest(ROOT / entry["p1_result_path"]) == entry["p1_result_sha256"]
    assert entry["p1_result_sha256"] == "abcc814dce1d268870d110aaa68775d0cccb5ff96aecfbdd0781e63a5a6bb174"
    assert entry["p1_candidate_report_path"].endswith("marker-v24-retry4/P1-run/candidate-report.json")
    assert digest(ROOT / entry["p1_candidate_report_path"]) == entry["p1_candidate_report_sha256"]
    assert entry["p1_candidate_report_sha256"] == entry["p1_result_sha256"]
    assert entry["retry3_p1_result_path"].endswith("P1_RETRY3_RESULT.json")
    assert entry["retry3_p1_result_sha256"] == "edf2ba744146fdcb6407b68d5765784f733b2f2e8739ecceedfd651edb372711"
    assert entry["retry3_p1_checkpoint_sha256"] == "70b9947bdaa78d5465f7cd2026a4bc00fd3805507551c002daf763e5dbc0b318"
    assert entry["retry3_p1_onnx_sha256"] == "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"
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
    assert digest(ROOT / entry["retry3_morphology_gap_path"]) == entry["retry3_morphology_gap_sha256"]
    assert entry["retry3_morphology_gap_sha256"] == "3b0e9981eb3d21787679f1df1151a3c0bc395ce7966e6c681c6de5755c3fb769"
    assert entry["retry3_morphology_gap_real_dev_projects"] == 120
    assert entry["retry3_morphology_gap_real_dev_failures"] == 0
    assert entry["retry3_morphology_gap_real_sealed_reads"] == 0
    assert entry["retry3_morphology_gap_real_dev_proposals"] == 1358010
    assert entry["retry3_morphology_gap_real_dev_positive_proposals"] == 9849
    assert entry["retry3_morphology_gap_real_dev_negative_below_threshold"] == 1337707
    assert entry["retry3_morphology_gap_real_dev_negative_above_threshold"] == 10454
    assert entry["retry3_morphology_gap_real_dev_runtime_ms"] == 444918.7276
    assert entry["retry3_morphology_gap_synthetic_negative_below_threshold"] == 233674
    assert entry["retry3_morphology_gap_synthetic_negative_above_threshold"] == 646
    assert entry["retry3_morphology_gap_real_to_synthetic_above_rate_ratio"] == 2.812662201375141
    assert entry["retry3_morphology_gap_case_level_output"] is False
    assert entry["retry3_morphology_gap_truth_rows_output"] is False
    assert entry["retry3_morphology_gap_pixel_output"] is False
    assert entry["retry3_morphology_gap_training_use"] is False
    assert entry["retry3_morphology_gap_candidate_selection"] is False
    assert entry["p1_checkpoint_sha256"] == "4d98a1de07282f734c6249c164400c237be9560370315b68c4729e9d15e5293c"
    assert entry["p1_onnx_sha256"] == "697fbcfb961e4c2af36a1a3d68cf5be874412b2939b03c42b59aaa82c4b0de96"
    assert entry["p1_true_positives"] == 1989
    assert entry["p1_false_positives"] == 138
    assert entry["p1_false_negatives"] == 15
    assert entry["p1_precision"] == 0.9351198871650211
    assert entry["p1_recall"] == 0.9925149700598802
    assert entry["p1_f1"] == 0.962962962962963
    assert entry["p1_onnx_parity_maximum_absolute_error"] == 4.76837158203125e-07
    assert entry["p1_opened_seal_sha256"] == "a262b0a46c2ef979a0d58591e454b6198cf550b658d6ae3e2f84d5277b276d14"
    assert entry["p1_result_seal_sha256"] == "373902ffa16d6ec2d23271c9a3f97fdca25a90b761a7965efcb19fd630141ee8"
    assert digest(ROOT / entry["retry4_diagnosis_path"]) == entry["retry4_diagnosis_sha256"]
    assert digest(ROOT / entry["retry4_generic_fp_diagnosis_path"]) == entry["retry4_generic_fp_diagnosis_sha256"]
    assert entry["retry4_accepted_false_positive_count"] == 138
    assert entry["retry4_accepted_false_positive_generic"] == 136
    assert entry["retry4_accepted_false_positive_artifact"] == 1
    assert entry["retry4_accepted_false_positive_topology_junction"] == 1
    assert entry["retry4_accepted_false_positive_topology_fragment"] == 0
    assert entry["retry4_topology_above_threshold_junction"] == 1
    assert entry["retry4_topology_capacity_junction"] == 8331
    assert entry["retry4_topology_above_threshold_fragment"] == 0
    assert entry["retry4_topology_capacity_fragment"] == 8049
    assert entry["retry4_generic_root_cause_connecting_line"] == 102
    assert entry["retry4_generic_root_cause_masked_context"] == 13
    assert entry["retry4_generic_root_cause_marker_field"] == 21
    assert entry["retry4_generic_root_cause_exhaustive_count"] == 136
    assert entry["p1_dev_gate_passed"] is False
    assert entry["negative_sampler_selected_index_sha256"] == "a81fb8127cda6819f1d0da318dc34ea2891dec5e3c6c767eb756af68bd2f869f"
    assert entry["negative_sampler_connector_anchor_fractions"] == [0.3333333333333333, 0.6666666666666666]
    assert entry["negative_sampler_connector_anchor_max_distance_px"] == 4.0
    assert entry["negative_sampler_connector_anchor_target_count"] == 3674
    assert entry["negative_sampler_connector_anchor_capacity"] == 3661
    assert entry["negative_sampler_connector_anchor_selected"] == 3661
    assert entry["negative_sampler_connector_anchor_selected_index_sha256"] == "40a14e3becc3f793a590cc2b37fdd92107a97f71f59c6fc1cf9d22d4d124116d"
    assert entry["negative_sampler_topology_radius_px"] == 16.0
    assert entry["negative_sampler_topology_capacity"] == {"topology_junction": 8331, "topology_fragment": 8049}
    assert entry["negative_sampler_topology_selected"] == {"topology_junction": 8331, "topology_fragment": 8049}
    assert entry["negative_sampler_topology_selected_index_sha256"] == "df24e495a76d485adcef07defd96ee723da3e029ededd5009e9c34e7ac58325d"
    assert entry["real_dev_result_path"].endswith("V24-RETRY3-REAL-DEV-STAGES.json")
    assert entry["real_dev_result_sha256"] == "1e471a179114e078f101edffcf04aea9e3b29ab72a3d31649695726465661c90"
    assert digest(ROOT / entry["real_dev_result_path"]) == entry["real_dev_result_sha256"]
    assert entry["real_dev_projects"] == 120
    assert entry["real_dev_successful_projects"] == 120
    assert entry["real_dev_failure_count"] == 0
    assert entry["real_dev_true_positives"] == 1103
    assert entry["real_dev_false_positives"] == 5978
    assert entry["real_dev_false_negatives"] == 901
    assert entry["real_dev_precision"] == 0.15576895918655556
    assert entry["real_dev_recall"] == 0.5503992015968064
    assert entry["real_dev_pre_nms_true_positives"] == 1126
    assert entry["real_dev_pre_nms_false_positives"] == 11273
    assert entry["real_dev_pre_nms_false_negatives"] == 878
    assert entry["real_dev_above_threshold_outputs"] == 12480
    assert entry["real_dev_above_threshold_true_positives"] == 1126
    assert entry["real_dev_above_threshold_false_positives"] == 11354
    assert entry["real_dev_final_outputs"] == 7081
    assert entry["real_dev_elapsed_ms"] == 436819.2499
    assert entry["real_dev_model_sha256"] == "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"
    assert entry["real_dev_case_level_output"] is False
    assert entry["real_dev_truth_rows_output"] is False
    assert entry["real_dev_pixel_output"] is False
    assert entry["real_dev_training_use"] is False
    assert entry["real_dev_candidate_selection"] is False
    assert entry["retry3_vs_retry2_precision_delta"] == 0.03542454589060041
    assert entry["retry3_vs_retry2_recall_delta"] == -0.04940119760479042
    assert entry["retry3_vs_retry2_false_positives_delta"] == -2808
    assert entry["retry3_vs_retry2_above_threshold_false_candidates_delta"] == -9920
    assert entry["spatial_morphology_diagnostic_required"] is False
    assert entry["retry2_dev_gate_passed"] is True
    assert entry["execution_blocker"] is None

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
