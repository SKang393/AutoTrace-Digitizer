# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations
import json
from pathlib import Path
from ml.markers.gate_seal import sha256_file, source_bundle_sha256
from ml.ocr.production_composition_v4.dataset import build_split, proposal_summary, split_fingerprint
from ml.ocr.production_composition_v4.protocol import AMBIGUITY_RECOGNIZER_ONNX_SHA256, REVISION, SPLITS, protocol_configuration
from ml.ocr.production_composition_v4.sealed_gate import EVALUATOR_SOURCE_PATHS

REPO_ROOT=Path(__file__).resolve().parents[4]; ROOT=REPO_ROOT/"ml/ocr/production_composition_v4"
def load(path:Path)->dict[str,object]: return json.loads(path.read_text(encoding="utf-8"))

def test_protocol_is_fresh_fail_closed_four_model_composition()->None:
    protocol=load(ROOT/"PROTOCOL.json"); expected=json.loads(json.dumps(protocol_configuration())); expected["split_generator_source_paths"]=protocol["split_generator_source_paths"]; expected["split_generator_source_bundle_sha256"]=protocol["split_generator_source_bundle_sha256"]
    assert protocol==expected and protocol["revision"]==REVISION
    assert protocol["models"]["ambiguity_specialist"]["onnx_sha256"]==AMBIGUITY_RECOGNIZER_ONNX_SHA256
    assert protocol["predecessor"]["fixture_bytes_reused"] is False
    assert protocol["production_approval"] is False and protocol["release_eligible"] is False

def test_fresh_splits_are_disjoint_complete_and_unopened()->None:
    validation=load(ROOT/"VALIDATION_SEAL.json"); public=load(ROOT/"SEALED_PUBLIC_TEST_SEAL.json"); fingerprints=set()
    for registration in SPLITS:
        scenes=build_split(registration.split); summary=proposal_summary(scenes); fingerprint=split_fingerprint(scenes); seal=validation if registration.split=="validation" else public
        assert summary["positive_proposal_count"]==summary["truth_region_count"] and summary=={key:seal[key] for key in summary}
        assert fingerprint==seal["split_fingerprint"] and fingerprint not in fingerprints; fingerprints.add(fingerprint)
        assert sha256_file(REPO_ROOT/seal["fixture_archive_path"])==seal["fixture_archive_sha256"]
    assert validation["validation_model_execution_count"]==0
    assert public["truth_hidden_from_model_execution_until_gate"] is True
    assert public["prior_public_sample_or_pixel_inspection_used"] is False

def test_gate_binds_all_transitive_sources_and_passed_component_hashes()->None:
    config=load(ROOT/"gates/sealed-public-v1.json")
    assert config["expected_evaluator_source_bundle_sha256"]==source_bundle_sha256(REPO_ROOT,EVALUATOR_SOURCE_PATHS)
    assert "ambiguity_recognizer_onnx_sha256" in config["expected_candidate_hash_keys"]
    assert config["production_approval"] is False and config["release_eligible"] is False

def test_consumed_predecessor_and_component_evidence_are_unchanged()->None:
    assert sha256_file(REPO_ROOT/"ml/ocr/production_composition_v3/VALIDATION_REPORT.json")=="905bb12948ce7bdcdba95f4940e9b1b5f97017da6586c808ff5c43e128049ea9"
    assert sha256_file(REPO_ROOT/"ml/ocr/ambiguity_context_classifier_v2/artifacts/P2-run/graph-ambiguity-line-context-v2-p2.onnx")==AMBIGUITY_RECOGNIZER_ONNX_SHA256
    assert not (ROOT/"VALIDATION_REPORT.json").exists() and not (ROOT/"PUBLIC_GATE_REPORT.json").exists()
