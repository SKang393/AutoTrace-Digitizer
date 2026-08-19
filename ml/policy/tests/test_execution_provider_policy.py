# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

from __future__ import annotations

import pytest

from ml.policy.execution_provider_policy import (
    ExecutionProviderPolicyError,
    select_execution_providers,
    session_options_for_split,
)


def test_train_and_dev_default_to_cpu_when_no_alternate_is_requested() -> None:
    for split in ("train", "dev"):
        assert select_execution_providers(
            split,
            available_providers=["CPUExecutionProvider"],
        ) == ("CPUExecutionProvider",)


def test_reviewed_discovered_directml_is_explicitly_selected_with_cpu_fallback() -> None:
    options = session_options_for_split(
        "dev",
        available_providers=["CPUExecutionProvider", "DmlExecutionProvider"],
        requested_provider="DmlExecutionProvider",
        provenance_reviewed=True,
    )

    assert options.providers == ("DmlExecutionProvider", "CPUExecutionProvider")
    assert options.graph_optimization == "ORT_ENABLE_ALL"


def test_any_installed_provider_can_be_selected_when_reviewed() -> None:
    assert select_execution_providers(
        "train",
        available_providers=["CPUExecutionProvider", "ExperimentalExecutionProvider"],
        requested_provider="ExperimentalExecutionProvider",
        provenance_reviewed=True,
    ) == ("ExperimentalExecutionProvider", "CPUExecutionProvider")


def test_non_cpu_provider_requires_reviewed_provenance() -> None:
    with pytest.raises(ExecutionProviderPolicyError, match="provenance_reviewed=True"):
        select_execution_providers(
            "train",
            available_providers=["CPUExecutionProvider", "DmlExecutionProvider"],
            requested_provider="DmlExecutionProvider",
        )


def test_sealed_rejects_non_cpu_even_when_discovered_and_reviewed() -> None:
    with pytest.raises(ExecutionProviderPolicyError, match="CPUExecutionProvider only"):
        session_options_for_split(
            "sealed",
            available_providers=["CPUExecutionProvider", "DmlExecutionProvider"],
            requested_provider="DmlExecutionProvider",
            provenance_reviewed=True,
        )


def test_cpu_fallback_and_discovery_are_required() -> None:
    with pytest.raises(ExecutionProviderPolicyError, match="CPUExecutionProvider"):
        select_execution_providers(
            "dev",
            available_providers=["DmlExecutionProvider"],
            requested_provider="DmlExecutionProvider",
            provenance_reviewed=True,
        )
    with pytest.raises(ExecutionProviderPolicyError, match="not discovered"):
        select_execution_providers(
            "dev",
            available_providers=["CPUExecutionProvider"],
            requested_provider="DmlExecutionProvider",
            provenance_reviewed=True,
        )
