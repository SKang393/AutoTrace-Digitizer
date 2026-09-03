# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Frozen V19 declarations for disjoint synthetic train-family coverage."""

from __future__ import annotations

TASK = "marker-center"
REVISION = "marker-center-train-family-v19"
CANDIDATE_ID = "P1"

V18_RESULT_PATH = "ml/markers/center/hard_positive_v18/P1_RESULT.json"
V18_RESULT_SHA256 = "f8aa3053b4d3da51060cd2c6739e80c43dbf037cdce466808e89c8fca65d57e9"
V13_MANIFEST_SHA256 = "771141bd5d22d2ffdb56e63f193c16c0775a0f47c02313ef67b06bbce9603126"
EVIDENCE_POLICY_PATH = "ml/policy/evidence-policy.json"
EVIDENCE_POLICY_SHA256 = "4dc18136c284b0b1805d3a3b22a9197ad06e6a41f4e43b4e1d4d9245b97e0aed"
MODEL_LICENSE = "Apache-2.0"

ACCEPTANCE_BAR = {"precision_minimum": 0.95, "recall_minimum": 0.95}
THRESHOLDS = (0.25, 0.40, 0.55, 0.70)
LABEL_POSITIVE_DISTANCE_PX = 3.0
FIXED_CONFIDENCE_THRESHOLD = 0.25
RUNTIME_CONTRACT = {
    "input": ["candidate_count", 3, 33, 33],
    "output": ["candidate_count", 4],
    "radius_minimum": 2.5,
    "radius_maximum": 8.0,
}

# These are aggregate properties measured from the private real corpus. They
# define synthetic coverage only; no private image, answer, or case identity is
# read by this revision.
TRAIN_FAMILY_SPECS = {
    "realrange_small_rgb_train": {
        "source_size_range": [360, 207, 640, 360],
        "color_modes": ["RGB"],
        "jpeg_quality_range": [55, 85],
        "resize_long_scale_range": [1.50, 1.50],
        "post_resize_text_height_px": [7, 12],
        "marker_diameter_px": [6, 25],
    },
    "realrange_median_rgba_train": {
        "source_size_range": [1338, 492, 1338, 492],
        "color_modes": ["RGBA"],
        "jpeg_quality_range": [70, 92],
        "resize_long_scale_range": [0.72, 0.72],
        "post_resize_text_height_px": [5, 9],
        "marker_diameter_px": [6, 25],
    },
    "realrange_wide_rgb_train": {
        "source_size_range": [2400, 1200, 4096, 3000],
        "color_modes": ["RGB"],
        "jpeg_quality_range": [45, 80],
        "resize_long_scale_range": [0.24, 0.40],
        "post_resize_text_height_px": [3, 7],
        "marker_diameter_px": [6, 25],
    },
    "realrange_large_rgba_train": {
        "source_size_range": [6352, 4484, 6352, 4484],
        "color_modes": ["RGBA"],
        "jpeg_quality_range": [35, 65],
        "resize_long_scale_range": [0.15, 0.15],
        "post_resize_text_height_px": [2, 5],
        "marker_diameter_px": [6, 25],
    },
}
TRAIN_VARIANTS_PER_FAMILY = 3
TRAIN_SEED_BASE = 2_609_000
DEV_SPLIT_ID = "marker-center-proposal-geometry-v13-dev"
SEALED_RUN_BUDGET = 1

__all__ = [
    "ACCEPTANCE_BAR", "CANDIDATE_ID", "DEV_SPLIT_ID", "EVIDENCE_POLICY_PATH",
    "EVIDENCE_POLICY_SHA256", "FIXED_CONFIDENCE_THRESHOLD", "LABEL_POSITIVE_DISTANCE_PX",
    "MODEL_LICENSE", "RUNTIME_CONTRACT", "REVISION", "SEALED_RUN_BUDGET", "TASK",
    "THRESHOLDS", "TRAIN_FAMILY_SPECS", "TRAIN_SEED_BASE", "TRAIN_VARIANTS_PER_FAMILY",
    "V13_MANIFEST_SHA256", "V18_RESULT_PATH", "V18_RESULT_SHA256",
]
