# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P1 training for the DB-objective detector defect class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_runner import REPO_ROOT, run_candidate
from .protocol import CANONICAL_OUTPUT


CONFIG_PATH = Path("ml/ocr/graph_text_db_objective_v5/training/p1.json")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_db_objective_v5/candidate_runner.py"),
    Path("ml/ocr/graph_text_db_objective_v5/dataset.py"),
    Path("ml/ocr/graph_text_db_objective_v5/losses.py"),
    Path("ml/ocr/graph_text_db_objective_v5/model.py"),
    Path("ml/ocr/graph_text_db_objective_v5/protocol.py"),
    Path("ml/ocr/graph_text_db_objective_v5/train_p1.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p1.py"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = run_candidate(
        candidate_id="P1",
        config_path=CONFIG_PATH,
        output_dir=(REPO_ROOT / arguments.output),
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())

