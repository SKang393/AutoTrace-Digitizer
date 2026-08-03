# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Command-line entry point for deterministic synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import PRESETS, generate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate license-clean synthetic single-case design graphs."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="smoke",
        help="Fixed generation matrix to render.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Dataset seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to the ignored ml/synthetic/datasets tree.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate_dataset(args.preset, args.seed, args.output)
    print(f"Generated {result.case_count} scenes in {result.output_directory}")
    print(
        "Sanity: "
        f"markers={result.marker_count}, "
        f"dividers={result.divider_count}, "
        f"hard_negatives={result.hard_negative_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
