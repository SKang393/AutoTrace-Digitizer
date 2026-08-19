# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authoritative evidence-policy decisions shared by model revisions."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).with_name("evidence-policy.json")
POLICY_REPOSITORY_PATH = "ml/policy/evidence-policy.json"


def _read_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    splits = policy.get("splits", {})
    if set(splits) != {"train", "dev", "sealed"}:
        raise ValueError("evidence policy must define train, dev, and sealed splits")
    if any(splits[name]["consumes_candidate_budget"] for name in ("train", "dev")):
        raise ValueError("train and dev must not consume candidate budget")
    if not splits["sealed"]["consumes_candidate_budget"]:
        raise ValueError("sealed must consume candidate budget")
    if policy["budget_accounting"]["consumption_event"] != "first_sealed_split_read":
        raise ValueError("candidate budget may be consumed only at first sealed split read")
    return policy


def load_evidence_policy() -> dict[str, Any]:
    """Return an independent copy of the checked authoritative policy."""

    return deepcopy(_read_policy())


def evidence_policy_reference() -> dict[str, object]:
    """Return the checksum-bound reference stored by revision protocols."""

    policy = _read_policy()
    return {
        "path": POLICY_REPOSITORY_PATH,
        "schema_version": policy["schema_version"],
        "policy_revision": policy["policy_revision"],
        "sha256": hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
    }


def split_rule(split: str) -> dict[str, Any]:
    """Return the authoritative rule for one split."""

    try:
        return deepcopy(_read_policy()["splits"][split])
    except KeyError as error:
        raise ValueError(f"unsupported evidence split: {split}") from error


def consumes_candidate_budget(split: str) -> bool:
    """Report whether reading the named split consumes a candidate."""

    return bool(split_rule(split)["consumes_candidate_budget"])


def classify_candidate_failure(*, sealed_read_started: bool) -> str:
    """Classify a failed run under the shared budget rule."""

    return "consumed" if sealed_read_started else "void"


__all__ = [
    "POLICY_PATH",
    "POLICY_REPOSITORY_PATH",
    "classify_candidate_failure",
    "consumes_candidate_budget",
    "evidence_policy_reference",
    "load_evidence_policy",
    "split_rule",
]
