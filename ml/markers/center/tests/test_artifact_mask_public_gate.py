# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

import pytest

from ml.markers.center.artifact_mask_public_gate import (
    PROHIBITED_STRUCTURE_KINDS,
    evaluate_public_gate,
)


def _row(fixture_id: str, *, arrow_shaft_hits: int = 0) -> dict[str, object]:
    hits = {kind: 0 for kind in PROHIBITED_STRUCTURE_KINDS}
    hits["arrow_shaft"] = arrow_shaft_hits
    return {
        "fixture_id": fixture_id,
        "expected_count": 4,
        "predicted_count": 4,
        "false_positive_count": 0,
        "false_negative_count": 0,
        "duplicate_count": 0,
        "prohibited_structure_hits": hits,
    }


def test_exact_public_fixtures_pass_with_frozen_taxonomy() -> None:
    result = evaluate_public_gate([_row("a"), _row("b"), _row("c")])

    assert result["status"] == "pass"
    assert result["fixture_count"] == 3
    assert result["exact_fixture_count"] == 3
    assert set(result["prohibited_structure_hits"]) == set(PROHIBITED_STRUCTURE_KINDS)


def test_arrow_shaft_hit_fails_the_public_gate() -> None:
    result = evaluate_public_gate([_row("a", arrow_shaft_hits=1), _row("b"), _row("c")])

    assert result["status"] == "fail"
    assert result["exact_fixture_count"] == 2
    assert result["prohibited_structure_hits"]["arrow_shaft"] == 1


def test_missing_taxonomy_and_duplicate_fixture_ids_are_rejected() -> None:
    missing_taxonomy = _row("a")
    missing_taxonomy["prohibited_structure_hits"].pop("arrow_shaft")
    with pytest.raises(ValueError, match="frozen taxonomy"):
        evaluate_public_gate([missing_taxonomy, _row("b"), _row("c")])

    with pytest.raises(ValueError, match="three unique fixtures"):
        evaluate_public_gate([_row("a"), _row("a"), _row("c")])
