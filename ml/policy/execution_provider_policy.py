# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Split-aware execution-provider selection without importing an inference runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ml.policy.evidence_policy import cpu_execution_provider, execution_provider_rule


class ExecutionProviderPolicyError(ValueError):
    """Raised when a provider request violates the authoritative split policy."""


@dataclass(frozen=True)
class SessionOptionsPolicy:
    split: str
    providers: tuple[str, ...]
    graph_optimization: str


def select_execution_providers(
    split: str,
    *,
    available_providers: Iterable[str],
    requested_provider: str | None = None,
    provenance_reviewed: bool = False,
) -> tuple[str, ...]:
    """Select an explicit provider with mandatory CPU fallback.

    ``available_providers`` is the runtime's discovered provider list.  A
    non-CPU provider is accepted only when explicitly requested, discovered,
    permitted by policy, and accompanied by reviewed provenance.  Sealed runs
    are always CPU-only.
    """

    rule = execution_provider_rule(split)
    available = tuple(str(provider) for provider in available_providers)
    if not available or len(set(available)) != len(available):
        raise ExecutionProviderPolicyError("available providers must be nonempty and unique")

    cpu = cpu_execution_provider()
    if cpu not in available:
        raise ExecutionProviderPolicyError("CPUExecutionProvider is required for fallback")

    requested = requested_provider or cpu
    if requested == cpu:
        return (cpu,)
    if split == "sealed":
        raise ExecutionProviderPolicyError(
            "sealed execution permits CPUExecutionProvider only"
        )
    if requested not in available:
        raise ExecutionProviderPolicyError(
            f"provider '{requested}' was not discovered by the runtime"
        )
    if not provenance_reviewed:
        raise ExecutionProviderPolicyError(
            f"provider '{requested}' requires provenance_reviewed=True"
        )
    return (requested, cpu)


def session_options_for_split(
    split: str,
    *,
    available_providers: Iterable[str],
    requested_provider: str | None = None,
    provenance_reviewed: bool = False,
) -> SessionOptionsPolicy:
    """Return provider order and graph-optimization policy for a split."""

    rule = execution_provider_rule(split)
    providers = select_execution_providers(
        split,
        available_providers=available_providers,
        requested_provider=requested_provider,
        provenance_reviewed=provenance_reviewed,
    )
    return SessionOptionsPolicy(
        split=split,
        providers=providers,
        graph_optimization=str(rule["graph_optimization"]),
    )


__all__ = [
    "ExecutionProviderPolicyError",
    "SessionOptionsPolicy",
    "select_execution_providers",
    "session_options_for_split",
]
