# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations
TASK="ocr-recognition"; REVISION="graph-ambiguity-source-group-v3"; PUBLIC_REVISION=f"{REVISION}-public-v1"; CANDIDATE_ID="P2"; SEED=20261531; IMAGE_SIZE=32
GLYPHS=("O","o","l","I"); COUNTS_PER_CLASS={"train":640,"validation":160,"sealed_public":240}
GATES={"validation_accuracy_minimum":0.97,"validation_macro_accuracy_minimum":0.97,"validation_per_class_accuracy_minimum":0.95,
       "sealed_accuracy_minimum":0.97,"sealed_macro_accuracy_minimum":0.97,"sealed_per_class_accuracy_minimum":0.95,
       "onnx_parity_maximum_absolute_error":0.00001,"onnx_argmax_mismatch_count":0,"provider":"CPUExecutionProvider"}
def protocol_configuration(*,runner_source_bundle_sha256:str)->dict[str,object]:
    return {"evidence_policy":"ml/policy/evidence-policy.json","schema":"graphreader.ocr-ambiguity-source-group-protocol.v1","task":TASK,"revision":REVISION,"status":"candidate_2_preregistered",
            "defect_class":"the public-passing line-box classifier was not trained on the exact multi-group production crop adapter and mapped lowercase o to uppercase O in V4 composition",
            "trigger_report_path":"ml/ocr/production_composition_v4/VALIDATION_REPORT.json","trigger_report_sha256":"075eb4cfee77591b8c2f16e3752a85364db261425ca477d99b26d940733a978e",
            "trigger_ambiguity_exact_match":0.5,"trigger_public_archive_opened":False,"prior_exposed_fixture_bytes_reused":False,
            "experiment_budget":3,"currently_preregistered_candidate":CANDIDATE_ID,"consumed_candidates":["P1"],
            "architecture":"source-group-profile-cnn-v1","isolated_change":"train new Apache-2.0 weights on fresh full-line crops transformed by the exact shared source-group adapter",
            "model_license":"Apache-2.0","classes":list(GLYPHS),
            "p1_result":{"path":"ml/ocr/ambiguity_source_group_classifier_v3/P1_RESULT.json","sha256":"ab9f6cc6ac7429caba08622029d1af2233c2713b4cfbc837c7dc729af13cf523","selection_accuracy":1.0,"onnx_parity_maximum_absolute_error":0.000030517578125,"public_archive_opened":False},
            "p2_isolated_change":"multiply exact P1 output logits by 0.125 during export with zero optimizer steps",
            "seed":SEED,"split_counts_per_class":COUNTS_PER_CLASS,"gates":GATES,
            "input_contract":{"name":"glyphs","shape":["batch",1,32,32],"dtype":"float32","source":"full-line crop plus target group index"},
            "output_contract":{"name":"logits","shape":["batch",4],"class_order":list(GLYPHS)},"runner_source_bundle_sha256":runner_source_bundle_sha256,
            "synthetic_only":True,"private_or_article_images":False,"chandler_included":False,"generalization_label_included":False,
            "manifest_created":False,"model_store_promoted":False,"production_approval":False,"release_eligible":False}
__all__=["CANDIDATE_ID","COUNTS_PER_CLASS","GATES","GLYPHS","IMAGE_SIZE","PUBLIC_REVISION","REVISION","SEED","TASK","protocol_configuration"]
