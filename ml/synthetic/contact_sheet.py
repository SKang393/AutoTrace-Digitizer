# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Visual overview rendering for deterministic dataset review."""

from __future__ import annotations

import math
from typing import Sequence

from PIL import Image, ImageDraw


def build_contact_sheet(
    images: Sequence[Image.Image],
    *,
    columns: int = 3,
    cell_size: tuple[int, int] = (400, 300),
    padding: int = 12,
) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required for a contact sheet.")
    if columns < 1:
        raise ValueError("Contact sheet columns must be positive.")

    cell_width, cell_height = cell_size
    rows = math.ceil(len(images) / columns)
    sheet_width = padding + columns * (cell_width + padding)
    sheet_height = padding + rows * (cell_height + padding)
    sheet = Image.new("L", (sheet_width, sheet_height), 238)
    draw = ImageDraw.Draw(sheet)

    for index, source in enumerate(images):
        row, column = divmod(index, columns)
        left = padding + column * (cell_width + padding)
        top = padding + row * (cell_height + padding)
        preview = source.convert("L")
        preview.thumbnail(
            (cell_width - 2, cell_height - 2),
            resample=Image.Resampling.LANCZOS,
            reducing_gap=2.0,
        )
        offset_x = left + (cell_width - preview.width) // 2
        offset_y = top + (cell_height - preview.height) // 2
        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            fill=255,
            outline=80,
            width=1,
        )
        sheet.paste(preview, (offset_x, offset_y))

    return sheet


__all__ = ["build_contact_sheet"]
