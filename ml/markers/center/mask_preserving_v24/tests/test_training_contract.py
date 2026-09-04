# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
import json
import hashlib
from pathlib import Path

from ml.markers.center.mask_preserving_v24.train_p1 import RUNNER_SOURCE_PATHS
from ml.markers.gate_seal import source_bundle_sha256
from ml.markers.center.real_range_generator_v1.negative_sampler import CONNECTOR_ENDPOINT_OFFSET_PX, TOPOLOGY_HARD_RADIUS_PX, TOPOLOGY_RADIUS_PX, TOPOLOGY_SAMPLER_RADIUS_PX

ROOT = Path(__file__).resolve().parents[5]

def test_training_contract_is_fixed_and_candidate_not_run():
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    assert config["seed"] == 20260903
    assert config["confidence_threshold"] == 0.25
    assert config["selection_thresholds"] == [0.40, 0.55, 0.70]
    assert config["optimizer_steps_expected"] == 10080
    assert config["optimizer_steps_maximum"] == 10080
    assert config["training_example_count_expected"] == 35838
    assert config["retry_count"] == 7
    assert config["negative_sampler"]["total_expected"] == 32580
    assert config["negative_sampler"]["source_sha256"] == "623ddb69cff4b6c0247d6389bbf803d6fcfe3b3eb9856fc9c83fdf2b469662ee"
    assert config["negative_sampler"]["selected_index_sha256"] == "d2e1c5f22ba8657b04031242c1528a165730f56b50ccc56fdd89eb2e0c01bf1c"
    assert config["negative_sampler"]["expected_capacities"] == {"artifact": 14477, "faint_low": 8598, "faint_p05": 5127, "generic": 176403, "hard_existing": 6012, "ocr_heavy": 20594}
    assert config["negative_sampler"]["topology"] == {"radius_px": 12.0, "input_audit_radius_px": 16.0, "expected_capacity": {"topology_junction": 4505, "topology_fragment": 4574}, "expected_selected": {"topology_junction": 4505, "topology_fragment": 4574}, "selected_index_sha256": "b160b1dcfd9b0e4af8653bc8126ab26af31532754fd74045dabaf7badc6c6bf5", "hard": {"radius_px": 4.0, "legacy_capacity": 6012, "expected_capacity": {"topology_junction": 417, "topology_fragment": 484}, "expected_selected": {"topology_junction": 417, "topology_fragment": 484}, "hard_training_total": 6856}}
    assert config["negative_sampler"]["topology"]["radius_px"] == TOPOLOGY_SAMPLER_RADIUS_PX
    assert config["negative_sampler"]["topology"]["input_audit_radius_px"] == TOPOLOGY_RADIUS_PX
    assert config["negative_sampler"]["topology"]["hard"]["radius_px"] == TOPOLOGY_HARD_RADIUS_PX
    assert config["hard_negative_example_count_expected"] == 6856
    assert sum(config["negative_sampler"]["quotas"].values()) == 32580
    assert config["sealed_runs"] == 0 and config["private_data"] is False
    assert config["real_dev_reads"] == 0 and config["real_sealed_reads"] == 0
    assert config["retry3_morphology_gap_sha256"] == "3b0e9981eb3d21787679f1df1151a3c0bc395ce7966e6c681c6de5755c3fb769"
    assert config["retry4_diagnosis_sha256"] == "a19745f7904c8ec316a78a4e220e3133fc5f77fa80f471ed5337976bdbb6594b"
    assert config["retry4_generic_fp_diagnosis_sha256"] == "24d86878dc335803b2aacd6bab5105496cbb2fb51734b4eb0d8ead4feea5d172"
    assert config["retry5_diagnosis_sha256"] == "8f38fd10be6130c34b05aa9544491f59c9c95d8b962bd0010f9dbdf287c8228a"
    assert config["retry5_generic_fp_diagnosis_sha256"] == "701f43ec266ae63689200610ea68d4e5a18b1017fd0951d88f496252ef1076d8"
    assert config["retry6_diagnosis_sha256"] == "34a3bbdf68cd049162b40964ad66c4bfe17cf0f46c306f969497002755e12b0e"
    assert config["negative_sampler"]["connector"] == {"endpoint_offset_px": 8.0, "max_distance_px": 4.0, "target_count": 3674, "expected_capacity": 3671, "expected_selected": 3671, "selected_index_sha256": "2d9b5b6ffa7e70c390a7aa38ffe851871e8a5cebac3831aac616f60a0e141c84", "generic_remainder_selected": 9409}
    assert config["negative_sampler"]["connector"]["endpoint_offset_px"] == CONNECTOR_ENDPOINT_OFFSET_PX

def test_runner_source_bundle_is_relative_and_present():
    assert all(not path.is_absolute() for path in RUNNER_SOURCE_PATHS)
    assert all((ROOT / path).is_file() for path in RUNNER_SOURCE_PATHS)
    config = json.loads((ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json").read_text())
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == config["revision"])
    assert entry["p1_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["status"] == "dev_passed_retry7_unconsumed"
    if entry["execution_authorized"]:
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
        ("retry5_diagnosis_path", "retry5_diagnosis_sha256"),
        ("retry5_generic_fp_diagnosis_path", "retry5_generic_fp_diagnosis_sha256"),
        ("retry6_diagnosis_path", "retry6_diagnosis_sha256"),
    ):
        assert digest(ROOT / config[path_key]) == config[hash_key]
    audit = json.loads((ROOT / config["generator_audit_path"]).read_text())
    assert audit["splits"]["train"]["aggregate_sha256"] == config["train_split_sha256"]
    assert audit["splits"]["dev"]["aggregate_sha256"] == config["dev_split_sha256"]
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == config["revision"])
    assert digest(config_path) == entry["candidate_config_sha256"]["P1"]
    assert entry["expected_runner_source_bundle_sha256"] == config["expected_runner_source_bundle_sha256"]
    if entry["execution_authorized"]:
        assert digest(ROOT / config["generator_audit_path"]) == config["generator_audit_sha256"]
        assert digest(ROOT / config["negative_audit_path"]) == config["negative_audit_sha256"]
        assert digest(ROOT / "ml/markers/center/real_range_generator_v1/generator.py") == entry["negative_generator_sha256"]
        assert digest(ROOT / "ml/markers/center/real_range_generator_v1/negative_proposal_audit.py") == entry["synthetic_negative_audit_source_sha256"]
        assert digest(ROOT / "ml/markers/center/real_range_generator_v1/AUDIT.json") == entry["negative_generator_audit_sha256"]
        assert digest(ROOT / "ml/markers/center/real_range_generator_v1/NEGATIVE_PROPOSAL_AUDIT.json") == entry["synthetic_negative_audit_sha256"]
    assert entry["synthetic_negative_proposal_count"] == 231211
    assert entry["morphology_diagnosis_sha256"] == config["morphology_diagnosis_sha256"]
    assert entry["morphology_gap_sha256"] == config["morphology_gap_sha256"]
    assert entry["status"] == "dev_passed_retry7_unconsumed"
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["real_dev_authorized"] is False
    assert entry["real_sealed_authorized"] is False
    assert entry["real_sealed_reads"] == 0
    assert entry["sealed_runs"] == 0
    assert entry["consumed_candidate_ids"] == []
    assert entry["dev_passed_candidate_ids"] == ["P1"]
    assert entry["candidate_consumed"] is False
    assert entry["p1_result_path"].endswith("P1_RETRY7_RESULT.json")
    assert digest(ROOT / entry["p1_result_path"]) == entry["p1_result_sha256"]
    assert entry["p1_result_sha256"] == "6fc74bc7e0aa6c36d7dd0aac51af014ad5875261f9e7a1cb113e13727287d9be"
    assert entry["p1_candidate_report_path"].endswith("marker-v24-retry7/P1-run/candidate-report.json")
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
    assert entry["retry4_p1_result_sha256"] == "abcc814dce1d268870d110aaa68775d0cccb5ff96aecfbdd0781e63a5a6bb174"
    assert entry["retry4_p1_checkpoint_sha256"] == "4d98a1de07282f734c6249c164400c237be9560370315b68c4729e9d15e5293c"
    assert entry["retry4_p1_onnx_sha256"] == "697fbcfb961e4c2af36a1a3d68cf5be874412b2939b03c42b59aaa82c4b0de96"
    assert entry["retry5_p1_checkpoint_sha256"] == "a7eaba24b5e65e19c97f303aaf1c5622e5a68a1dabbf44e1df11386c15489832"
    assert entry["retry5_p1_onnx_sha256"] == "d3445f0b1bf0e97a98942133d45341cae75548887be853743e887832cacad7bd"
    assert entry["p1_checkpoint_sha256"] == "a66085d55d9d361a9d98db6105e3dcf2269dbff103cc542e91ff9fbf4fd0d350"
    assert entry["p1_onnx_sha256"] == "7932b008a9c4372c832215f2f8732c59c59012a25aa4ad2d12cfeaed404bbe3c"
    assert entry["p1_true_positives"] == 1991
    assert entry["p1_false_positives"] == 46
    assert entry["p1_false_negatives"] == 13
    assert entry["p1_precision"] == 0.9774177712322042
    assert entry["p1_recall"] == 0.9935129740518962
    assert entry["p1_f1"] == 0.9853996535511013
    assert entry["p1_prohibited_structure_hits"] == 0
    assert entry["p1_onnx_parity_maximum_absolute_error"] == 4.76837158203125e-07
    assert entry["retry5_p1_opened_seal_sha256"] == "db3c4ded697e108a7af55d3eada605c6bbe3cc0dd17775727fc401c885c41386"
    assert entry["retry5_p1_result_seal_sha256"] == "117bb7a12b5b742a47eb8a3fb3cd8e5899692ba1ec5119a48d5c0432e21a7c40"
    assert entry["p1_opened_seal_sha256"] == "3df9789fe00bed0204b102129045f4feb8e705efca2fe12920ba8745b215b86f"
    assert entry["p1_result_seal_sha256"] == "74badeb19c9d0acad23b5cf8ca8759fe55c3275a4deaec70fb71f45ed3a714c6"
    assert digest(ROOT / entry["retry6_diagnosis_path"]) == entry["retry6_diagnosis_sha256"]
    assert entry["retry6_accepted_false_positive_generic"] == 85
    assert entry["retry6_accepted_false_positive_topology_junction"] == 1
    assert entry["retry6_accepted_false_positive_connector_anchor"] == 0
    assert entry["retry6_accepted_false_positive_topology_fragment"] == 0
    assert entry["retry6_above_threshold_generic"] == 2171
    assert entry["retry6_above_threshold_artifact"] == 98
    assert entry["retry6_above_threshold_connector_anchor"] == 33
    assert entry["retry6_above_threshold_topology_junction"] == 8
    assert entry["retry6_above_threshold_topology_fragment"] == 3
    assert entry["retry6_prohibited_hit_kind"] == "topology_junction"
    assert entry["retry6_prohibited_hit_source"] == "topology_junction"
    assert entry["retry6_prohibited_hit_confidence"] == 0.3208221197128296
    assert entry["retry6_prohibited_hit_distance_px"] == 1.9674727110252526
    assert entry["retry6_diagnosis_topology_input_audit_radius_px"] == 16.0
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
    assert digest(ROOT / entry["retry5_diagnosis_path"]) == entry["retry5_diagnosis_sha256"]
    assert digest(ROOT / entry["retry5_generic_fp_diagnosis_path"]) == entry["retry5_generic_fp_diagnosis_sha256"]
    assert entry["retry5_accepted_false_positive_count"] == 184
    assert entry["retry5_accepted_false_positive_generic"] == 183
    assert entry["retry5_accepted_false_positive_topology_fragment"] == 1
    assert entry["retry5_accepted_false_positive_topology_junction"] == 0
    assert entry["retry5_accepted_false_positive_connector_anchor"] == 0
    assert entry["retry5_above_threshold_generic"] == 4818
    assert entry["retry5_above_threshold_artifact"] == 169
    assert entry["retry5_above_threshold_connector_anchor"] == 4
    assert entry["retry5_above_threshold_ocr"] == 2
    assert entry["retry5_above_threshold_topology_fragment"] == 1
    assert entry["retry5_above_threshold_topology_junction"] == 0
    assert entry["retry5_generic_root_cause_near_connecting_line"] == 115
    assert entry["retry5_generic_root_cause_masked_context"] == 23
    assert entry["retry5_generic_root_cause_marker_field"] == 45
    assert entry["retry5_generic_root_cause_exhaustive_count"] == 183
    assert entry["p1_dev_gate_passed"] is True
    assert entry["negative_sampler_selected_index_sha256"] == "d2e1c5f22ba8657b04031242c1528a165730f56b50ccc56fdd89eb2e0c01bf1c"
    assert entry["negative_sampler_connector_endpoint_offset_px"] == 8.0
    assert entry["negative_sampler_connector_anchor_max_distance_px"] == 4.0
    assert entry["negative_sampler_connector_anchor_target_count"] == 3674
    assert entry["negative_sampler_connector_anchor_capacity"] == 3671
    assert entry["negative_sampler_connector_anchor_selected"] == 3671
    assert entry["negative_sampler_connector_anchor_selected_index_sha256"] == "2d9b5b6ffa7e70c390a7aa38ffe851871e8a5cebac3831aac616f60a0e141c84"
    assert entry["negative_sampler_topology_radius_px"] == 12.0
    assert entry["negative_sampler_topology_input_audit_radius_px"] == 16.0
    assert entry["negative_sampler_topology_capacity"] == {"topology_junction": 4505, "topology_fragment": 4574}
    assert entry["negative_sampler_topology_selected"] == {"topology_junction": 4505, "topology_fragment": 4574}
    assert entry["negative_sampler_topology_selected_index_sha256"] == "b160b1dcfd9b0e4af8653bc8126ab26af31532754fd74045dabaf7badc6c6bf5"
    assert entry["negative_sampler_topology_hard_radius_px"] == 4.0
    assert entry["negative_sampler_topology_hard_legacy_capacity"] == 6012
    assert entry["negative_sampler_topology_hard_capacity"] == {"topology_junction": 417, "topology_fragment": 484}
    assert entry["negative_sampler_topology_hard_selected"] == {"topology_junction": 417, "topology_fragment": 484}
    assert entry["negative_sampler_hard_training_total"] == 6856
    assert entry["negative_sampler_generic_remainder_selected"] == 9409
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

def test_retry7_records_unconsumed_synthetic_dev_pass():
    result_path = ROOT / "ml/markers/center/mask_preserving_v24/P1_RETRY7_RESULT.json"
    result = json.loads(result_path.read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    ledger = json.loads((ROOT / "ml/markers/training-budgets/production-repair-v1.json").read_text())
    entry = next(item for item in ledger["revisions"] if item["revision"] == result["revision"])
    assert result["status"] == "dev_passed"
    assert result["synthetic_only"] is True
    assert result["private_data"] is False
    assert result["checkpoint_sha256"] == "a66085d55d9d361a9d98db6105e3dcf2269dbff103cc542e91ff9fbf4fd0d350"
    assert result["onnx_sha256"] == "7932b008a9c4372c832215f2f8732c59c59012a25aa4ad2d12cfeaed404bbe3c"
    assert result["selected"] == {
        "duplicate_count": 0,
        "f1": 0.9853996535511013,
        "false_negatives": 13,
        "false_positives": 46,
        "precision": 0.9774177712322042,
        "prohibited_structure_hits": 0,
        "proposal_recall": 1.0,
        "proposal_true_positives": 2004,
        "recall": 0.9935129740518962,
        "scene_count": 167,
        "threshold": 0.25,
        "true_positives": 1991,
    }
    assert result["optimizer_steps"] == 10080
    assert result["hard_negative_example_count"] == 6856
    assert result["onnx_parity_maximum_absolute_error"] == 4.76837158203125e-07
    assert result["elapsed_ms"] == 1468203.664
    assert result["real_dev_reads"] == 0
    assert result["real_sealed_reads"] == 0
    assert result["sealed_runs"] == 0
    assert digest(result_path) == "6fc74bc7e0aa6c36d7dd0aac51af014ad5875261f9e7a1cb113e13727287d9be"
    assert entry["status"] == "dev_passed_retry7_unconsumed"
    assert entry["dev_passed_candidate_ids"] == ["P1"]
    assert entry["consumed_candidate_ids"] == []
    assert entry["candidate_consumed"] is False
    assert entry["execution_authorized"] is False
    assert entry["authorized_candidate_id"] is None
    assert entry["real_dev_authorized"] is False
    assert entry["real_dev_reads"] == 0
    assert entry["real_dev_gate_passed"] is False
    assert entry["real_sealed_authorized"] is False
    assert entry["real_sealed_reads"] == 0
    assert entry["sealed_runs"] == 0
    assert entry["public_gate_authorized"] is False
    assert entry["public_gate_evaluations"] == 0
    assert entry["production_approval"] is False
    assert entry["release_eligible"] is False
    assert entry["p1_result_path"].endswith("P1_RETRY7_RESULT.json")
    assert entry["p1_result_sha256"] == "6fc74bc7e0aa6c36d7dd0aac51af014ad5875261f9e7a1cb113e13727287d9be"
    assert digest(ROOT / entry["p1_result_path"]) == entry["p1_result_sha256"]
    assert entry["p1_checkpoint_sha256"] == "a66085d55d9d361a9d98db6105e3dcf2269dbff103cc542e91ff9fbf4fd0d350"
    assert entry["p1_onnx_sha256"] == "7932b008a9c4372c832215f2f8732c59c59012a25aa4ad2d12cfeaed404bbe3c"
    assert entry["p1_opened_seal_sha256"] == "3df9789fe00bed0204b102129045f4feb8e705efca2fe12920ba8745b215b86f"
    assert entry["p1_result_seal_sha256"] == "74badeb19c9d0acad23b5cf8ca8759fe55c3275a4deaec70fb71f45ed3a714c6"
    assert entry["retry6_p1_result_sha256"] == "610487133e59a71b457c261cbffa0af9d64ddbdac96e5e5697b3f411a761c3a7"
    assert entry["retry6_p1_checkpoint_sha256"] == "e23503d79ca58c535fcdcff5cf344d87205e3e2ad901c208d7a4049dc530d5e8"
    assert entry["retry6_p1_onnx_sha256"] == "31d473d6c24bf21edc1cbfb25f7da35eabfed7cbf8afc13bf52bef23d06bfeb9"
    diagnosis_path = ROOT / "ml/markers/center/mask_preserving_v24/diagnostics/V24_RETRY7_DIAGNOSIS.json"
    diagnosis = json.loads(diagnosis_path.read_text())
    assert digest(diagnosis_path) == "1761fd27f0cd1aa9e6a1e3b2b8f0d3c4fa84cb5195dd8c72cea3f17d042683ac"
    assert entry["retry7_diagnosis_path"].endswith("V24_RETRY7_DIAGNOSIS.json")
    assert entry["retry7_diagnosis_sha256"] == "1761fd27f0cd1aa9e6a1e3b2b8f0d3c4fa84cb5195dd8c72cea3f17d042683ac"
    assert digest(ROOT / entry["retry7_diagnosis_path"]) == entry["retry7_diagnosis_sha256"]
    assert diagnosis["scope"] == {
        "case_ids_or_pixels_emitted": False,
        "label_positive_distance_px": 3.0,
        "optimizer_steps": 0,
        "private_data": False,
        "real_dev_reads": 0,
        "real_sealed_reads": 0,
        "retry_mode": "retry7",
        "scene_count": 167,
        "split": "real-range-generator-v1-dev",
        "synthetic_only": True,
        "threshold": 0.25,
        "truth_count": 2004,
    }
    assert diagnosis["fixed_threshold_metrics"] == {
        "accepted": 2037,
        "accepted_false_positive_attribution": {"generic": 46},
        "false_negatives": 13,
        "false_positives": 46,
        "precision": 0.9774177712322042,
        "prohibited_structure_hits": 0,
        "recall": 0.9935129740518962,
        "true_positives": 1991,
    }
    assert diagnosis["prohibited_hit_attribution"] == {
        "by_prohibited_kind": {},
        "by_source_group": {},
        "total": 0,
    }
    assert entry["retry7_diagnosis_scene_count"] == 167
    assert entry["retry7_diagnosis_threshold"] == 0.25
    assert entry["retry7_diagnosis_optimizer_steps"] == 0
    assert entry["retry7_diagnosis_private_data"] is False
    assert entry["retry7_diagnosis_real_dev_reads"] == 0
    assert entry["retry7_diagnosis_real_sealed_reads"] == 0
    assert entry["retry7_diagnosis_sealed_runs"] == 0
    assert entry["retry7_diagnosis_threshold_change_proposed"] is False
    assert entry["retry7_accepted_false_positive_generic"] == 46
    assert entry["retry7_accepted_false_positive_topology_junction"] == 0
    assert entry["retry7_accepted_false_positive_topology_fragment"] == 0
    assert entry["retry7_accepted_false_positive_connector_anchor"] == 0
    assert entry["retry7_prohibited_structure_hits"] == 0
    assert entry["p1_runner_source_bundle_sha256"] == "f884c1cbaa51ff8a0a859cf89662c9bba3dcc7c94a0f2d920184ddbcdab68951"
