# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""V38 dataset facade retaining V37 scenes, tiles, and full-box targets."""

from ml.ocr.degradation_coverage_detector_v37.dataset import (
    build_base_train_split,
    build_split,
    build_tiles,
    split_fingerprint,
    to_arrays,
)

__all__ = ["build_base_train_split", "build_split", "build_tiles", "split_fingerprint", "to_arrays"]
