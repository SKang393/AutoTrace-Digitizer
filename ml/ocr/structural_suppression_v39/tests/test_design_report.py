# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

import json
from pathlib import Path


def test_v39_is_closed_before_candidate_creation() -> None:
    report = json.loads(
        (Path(__file__).parents[1] / "BLOCKED_DESIGN_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["status"] == "blocked_before_candidate_creation"
    assert report["candidate_created"] is False
    assert report["candidate_startable"] is False
    assert report["public_or_sealed_reads"] == 0
    assert report["real_reads"] == 0
    assert report["private_or_article_reads"] == 0
    assert report["decision"]["no_candidate_created"] is True
