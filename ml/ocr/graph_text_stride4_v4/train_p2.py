# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use P2 training for the resize-convolution decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_runner import REPO_ROOT, run_candidate
from .model_p2 import Stride4ResizeConvTextRegionNet
from .protocol import SEED


CANDIDATE_ID = "P2"
CONFIG_PATH = Path("ml/ocr/graph_text_stride4_v4/training/p2.json")
CANONICAL_OUTPUT = Path("ml/ocr/graph_text_stride4_v4/artifacts/P2-run")
RUNNER_SOURCE_PATHS = (
    Path("ml/ocr/graph_text_stride4_v4/candidate_runner.py"),
    Path("ml/ocr/graph_text_stride4_v4/dataset.py"),
    Path("ml/ocr/graph_text_stride4_v4/model_p2.py"),
    Path("ml/ocr/graph_text_stride4_v4/protocol.py"),
    Path("ml/ocr/graph_text_stride4_v4/train_p2.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/dataset.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p1.py"),
    Path("ml/ocr/graph_text_ignore_band_v3/train_p3.py"),
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
        model_factory=lambda: Stride4ResizeConvTextRegionNet(seed=SEED),
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
