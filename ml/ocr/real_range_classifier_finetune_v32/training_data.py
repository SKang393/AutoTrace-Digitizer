# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Training-only proposal tensors from the frozen V32 train split."""

from .dataset import proposal_examples


def training_examples():
    values, labels, evidence = proposal_examples("train")
    return values, labels, evidence
