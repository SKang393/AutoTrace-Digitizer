# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .dataset import REPO_ROOT, write_freeze

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("ml/ocr/ambiguity_context_classifier_v2/artifacts/split-freeze"))
    args = parser.parse_args()
    print(json.dumps(write_freeze(REPO_ROOT / args.output_root), indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
