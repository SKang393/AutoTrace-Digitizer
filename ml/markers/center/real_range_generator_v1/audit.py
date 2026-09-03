# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""CLI for the aggregate-only synthetic range audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .generator import audit

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(audit(), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(encoded.encode("utf-8"))
    print(encoded, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
