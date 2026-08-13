# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen protocol for the project-owned ambiguity glyph classifier."""

from __future__ import annotations


TASK = "ocr-recognition"
REVISION = "graph-ambiguity-glyph-classifier-v1"
PUBLIC_REVISION = f"{REVISION}-public-v1"
CANDIDATE_ID = "P1"
EXPERIMENT_BUDGET = 3
SEED = 20261317
GLYPHS = ("O", "o", "l", "I")
IMAGE_SIZE = 24
COUNTS_PER_CLASS = {"train": 640, "validation": 160, "sealed_public": 240}
GATES = {
    "validation_accuracy_minimum": 0.97,
    "validation_macro_accuracy_minimum": 0.97,
    "validation_per_class_accuracy_minimum": 0.95,
    "sealed_accuracy_minimum": 0.97,
    "sealed_macro_accuracy_minimum": 0.97,
    "sealed_per_class_accuracy_minimum": 0.95,
    "onnx_parity_maximum_absolute_error": 0.00001,
    "provider": "CPUExecutionProvider",
}


def protocol_configuration(*, runner_source_bundle_sha256: str) -> dict[str, object]:
    return {
        "schema": "graphreader.ocr-ambiguity-glyph-protocol.v1",
        "task": TASK,
        "revision": REVISION,
        "status": "candidate_1_preregistered",
        "defect_class": (
            "the exact official recognizer collapses O/o/l/I identity on isolated source groups even after "
            "conservative spacing preserves all non-ambiguity text"
        ),
        "trigger_evidence": {
            "path": "ml/ocr/official_recognition_spacing_v3/P1_RESULT.json",
            "sha256": "5e92faebb3cb1d40bf4e948efc2b39f44ee82219c3393f4fde8d96088c6ea1f7",
            "selection_ambiguity_exact_match": 0.0,
            "selection_ambiguity_case_count": 32,
            "public_archive_opened": False,
        },
        "experiment_budget": EXPERIMENT_BUDGET,
        "currently_preregistered_candidate": CANDIDATE_ID,
        "consumed_candidates": [],
        "architecture": "compact-ambiguity-glyph-cnn-v1",
        "isolated_change": (
            "train a project-owned four-class O/o/l/I glyph classifier on fresh procedural Noto crops; "
            "do not use truth, role, graph position, word lists, private images, Chandler, or prior exposed pixels"
        ),
        "model_license": "Apache-2.0",
        "classes": list(GLYPHS),
        "input_contract": {"name": "glyphs", "shape": ["batch", 1, IMAGE_SIZE, IMAGE_SIZE], "dtype": "float32", "range": [0, 1]},
        "output_contract": {"name": "logits", "shape": ["batch", len(GLYPHS)], "class_order": list(GLYPHS)},
        "seed": SEED,
        "split_counts_per_class": COUNTS_PER_CLASS,
        "gates": GATES,
        "runner_source_bundle_sha256": runner_source_bundle_sha256,
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "generalization_label_included": False,
        "manifest_created": False,
        "model_store_promoted": False,
        "production_approval": False,
        "release_eligible": False,
    }


__all__ = [
    "CANDIDATE_ID", "COUNTS_PER_CLASS", "EXPERIMENT_BUDGET", "GATES", "GLYPHS",
    "IMAGE_SIZE", "PUBLIC_REVISION", "REVISION", "SEED", "TASK", "protocol_configuration",
]

