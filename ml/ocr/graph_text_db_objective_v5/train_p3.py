# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P3 training with explicit ignored-boundary negative supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_runner_p3 import REPO_ROOT, run_candidate


CANDIDATE_ID = "P3"
CONFIG_PATH = Path("ml/ocr/graph_text_db_objective_v5/training/p3.json")
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_db_objective_v5/artifacts/P3-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_db_objective_v5/candidate_runner_p3.py"),
    Path("ml/ocr/graph_text_db_objective_v5/dataset.py"),
    Path("ml/ocr/graph_text_db_objective_v5/losses.py"),
    Path("ml/ocr/graph_text_db_objective_v5/losses_p3.py"),
    Path("ml/ocr/graph_text_db_objective_v5/model.py"),
    Path("ml/ocr/graph_text_db_objective_v5/protocol.py"),
    Path("ml/ocr/graph_text_db_objective_v5/train_p3.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p1.py"),
    Path("ml/ocr/official_bakeoff/structure_consensus_evaluate.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def train_candidate(output_dir: Path) -> dict[str, object]:
    return run_candidate(
        candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH,
        output_dir=output_dir,
        runner_source_paths=RUNNER_SOURCE_PATHS,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    arguments = parser.parse_args()
    report = train_candidate(REPO_ROOT / arguments.output)
    print(
        json.dumps(
            {
                "candidate_id": report["candidate_id"],
                "status": report["status"],
                "selection_gate_passed": report["selection_gate_passed"],
                "probability_contract_passed": report["probability_contract_passed"],
                "onnx_parity_passed": report["onnx_parity_passed"],
                "selection_metrics": {
                    key: value
                    for key, value in report["selection_metrics"].items()
                    if key != "records"
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "selected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
