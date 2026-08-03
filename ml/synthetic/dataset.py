# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic dataset assembly, split manifests, and sanity checks."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image

from .contact_sheet import build_contact_sheet
from .io import sha256, write_csv, write_json, write_png
from .renderer import render_scene
from .schema import validate_scene
from .templates import (
    FILL_STATES,
    LINE_STYLES,
    MARKER_SHAPES,
    SCENE_FEATURES,
    build_scene,
)


HARD_NEGATIVE_KINDS = (
    "marker_like_letter",
    "arrowhead",
    "phase_line_intersection",
    "tick_mark",
    "dotted_divider_segment",
    "legend_symbol",
    "bracket",
    "isolated_punctuation",
    "line_endpoint",
)
REQUIRED_TEXT_ROLES = (
    "annotation",
    "axis_title",
    "legend_text",
    "participant",
    "phase_heading",
    "x_tick",
    "y_tick",
)
FAMILY_AXES = ("renderer", "font", "degradation", "template", "marker")
CSV_FIELDS = (
    "scene_id",
    "panel_id",
    "participant",
    "series_id",
    "point_id",
    "observation_index",
    "printed_x_value",
    "estimated_x_value",
    "x_confidence",
    "x_value",
    "y_value",
    "phase",
    "original_pixel_x",
    "original_pixel_y",
    "shape",
    "fill",
)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    design: str
    renderer_family: str
    panel_count: int
    session_count: int
    features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetResult:
    output_directory: Path
    case_count: int
    marker_count: int
    divider_count: int
    hard_negative_count: int


PRESETS: dict[str, tuple[CaseSpec, ...]] = {
    "smoke": (
        CaseSpec("ab", "vector_clean", 1, 2, ("missing_sessions",)),
        CaseSpec("aba", "print_monochrome", 1, 18, ("blank_phase_gaps",)),
        CaseSpec("abab", "vector_clean", 1, 24),
        CaseSpec(
            "multiple_baseline",
            "print_monochrome",
            3,
            20,
            ("missing_sessions",),
        ),
        CaseSpec(
            "multiple_probe",
            "scan_rough",
            2,
            30,
            ("sparse_probes",),
        ),
        CaseSpec(
            "alternating_treatments",
            "scan_rough",
            1,
            36,
            ("irregular_spacing",),
        ),
        CaseSpec("changing_criterion", "scan_rough", 6, 24),
        CaseSpec("maintenance", "hand_drawn", 1, 28),
        CaseSpec(
            "generalization",
            "hand_drawn",
            1,
            32,
            ("sparse_probes",),
        ),
        CaseSpec(
            "staggered_starts",
            "hand_drawn",
            3,
            40,
            ("irregular_spacing",),
        ),
        CaseSpec(
            "shared_baseline",
            "hand_drawn",
            1,
            100,
            ("blank_phase_gaps",),
        ),
    ),
}


_SYMBOLS = {
    ("circle", "filled"): "●",
    ("circle", "open"): "○",
    ("square", "filled"): "■",
    ("square", "open"): "□",
    ("triangle_up", "filled"): "▲",
    ("triangle_up", "open"): "△",
    ("triangle_down", "filled"): "▼",
    ("triangle_down", "open"): "▽",
    ("diamond", "filled"): "◆",
    ("diamond", "open"): "◇",
    ("star", "filled"): "★",
    ("star", "open"): "☆",
    ("asterisk", "filled"): "✱",
    ("asterisk", "open"): "✱",
    ("cross", "filled"): "✚",
    ("cross", "open"): "✚",
    ("other", "filled"): "⬟",
    ("other", "open"): "⬡",
}


def generate_dataset(
    preset: str,
    seed: int,
    output_directory: Path | None = None,
) -> DatasetResult:
    """Generate one fixed preset and fail if its perfect-label checks drift."""

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Expected one of {sorted(PRESETS)}.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    destination = (
        Path(output_directory).resolve()
        if output_directory is not None
        else Path(__file__).resolve().parent / "datasets" / f"{preset}-{seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)

    scenes = _build_scenes(PRESETS[preset], seed)
    case_entries: list[dict[str, Any]] = []
    rendered_images: list[Image.Image] = []
    output_files: list[Path] = []
    all_annotations: list[Mapping[str, Any]] = []

    for index, scene in enumerate(scenes):
        scene_id = str(scene["scene_id"])
        image, annotation, marker_mask = render_scene(scene)
        case_metrics = _validate_rendered_case(scene, annotation, marker_mask)
        rows = list(_csv_rows(scene, annotation))

        scene_path = destination / "scenes" / f"{scene_id}.json"
        image_path = destination / "images" / f"{scene_id}.png"
        annotation_path = destination / "annotations" / f"{scene_id}.json"
        mask_path = destination / "masks" / f"{scene_id}.png"
        csv_path = destination / "tables" / f"{scene_id}.csv"

        payloads = {
            scene_path: write_json(scene_path, scene),
            image_path: write_png(image_path, image),
            annotation_path: write_json(annotation_path, annotation),
            mask_path: write_png(mask_path, marker_mask),
            csv_path: write_csv(csv_path, CSV_FIELDS, rows),
        }
        output_files.extend(payloads)
        rendered_images.append(image.copy())
        all_annotations.append(annotation)

        split = _scene_split(scene)
        case_entries.append(
            {
                "case_index": index,
                "scene_id": scene_id,
                "seed": scene["seed"],
                "design": scene["design"],
                "split": split,
                "families": _family_keys(scene),
                "features": sorted(scene.get("layout", {}).get("features", [])),
                "files": {
                    path.relative_to(destination).as_posix(): sha256(payload)
                    for path, payload in sorted(
                        payloads.items(), key=lambda item: item[0].as_posix()
                    )
                },
                "metrics": case_metrics,
            }
        )

    split_payloads = _split_manifests(case_entries)
    for split, payload in split_payloads.items():
        path = destination / "splits" / f"{split}.json"
        write_json(path, payload)
        output_files.append(path)

    sanity = _dataset_sanity(scenes, all_annotations, case_entries)
    sanity_path = destination / "sanity-report.json"
    write_json(sanity_path, sanity)
    output_files.append(sanity_path)

    contact_sheet = build_contact_sheet(rendered_images)
    contact_path = destination / "contact-sheet.png"
    write_png(contact_path, contact_sheet)
    output_files.append(contact_path)

    seed_manifest = {
        "manifest_version": 1,
        "generator_version": "0.1.0",
        "preset": preset,
        "dataset_seed": seed,
        "renderer_environment": {
            "pillow_version": package_version("Pillow"),
            "font_policy": "installed_or_user_supplied_only",
            "font_files_bundled": False,
        },
        "cases": case_entries,
        "artifact_sha256": {
            path.relative_to(destination).as_posix(): sha256(path.read_bytes())
            for path in sorted(output_files, key=lambda item: item.as_posix())
        },
        "sanity_passed": sanity["passed"],
    }
    manifest_path = destination / "seed-manifest.json"
    write_json(manifest_path, seed_manifest)

    return DatasetResult(
        output_directory=destination,
        case_count=len(scenes),
        marker_count=int(sanity["counts"]["markers"]),
        divider_count=int(sanity["counts"]["dividers"]),
        hard_negative_count=int(sanity["counts"]["hard_negatives"]),
    )


def _build_scenes(specs: Sequence[CaseSpec], dataset_seed: int) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    style_catalog = [
        (shape, fill)
        for shape in MARKER_SHAPES
        for fill in FILL_STATES
    ]
    style_index = 0
    line_index = 0

    for case_index, spec in enumerate(specs):
        case_seed = dataset_seed * 100 + case_index
        scene = build_scene(
            spec.design,
            case_seed,
            spec.renderer_family,
            spec.panel_count,
            session_count=spec.session_count,
            features=spec.features,
        )
        for panel in scene["panels"]:
            points_by_series: dict[str, list[dict[str, Any]]] = {}
            for point in panel["points"]:
                points_by_series.setdefault(str(point["series_id"]), []).append(point)
            for series in panel["series"]:
                shape, fill = style_catalog[style_index % len(style_catalog)]
                style_index += 1
                line_style = LINE_STYLES[line_index % len(LINE_STYLES)]
                line_index += 1
                series["shape"] = shape
                series["fill"] = fill
                series["symbol"] = _symbol(shape, fill)
                series["line_style"] = line_style
                for point in points_by_series.get(str(series["series_id"]), []):
                    point["shape"] = shape
                    point["fill"] = fill
                    # The renderer emits the authoritative binary mask for the
                    # reassigned style. Avoid retaining a stale template polygon.
                    point["mask"] = None
        validate_scene(scene)
        scenes.append(scene)

    if style_index < len(style_catalog):
        raise AssertionError(
            f"Smoke matrix exposes {style_index} series but needs "
            f"{len(style_catalog)} for complete marker style coverage."
        )
    return scenes


def _symbol(shape: str, fill: str) -> str:
    if fill == "degraded":
        return _SYMBOLS.get((shape, "open"), "◇")
    return _SYMBOLS.get((shape, fill), "◇")


def _csv_rows(
    scene: Mapping[str, Any],
    annotation: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    rendered_centers = {
        str(marker.get("point_id", marker.get("marker_id"))): marker["center"]
        for marker in _annotation_records(annotation, "markers")
    }
    for panel in scene["panels"]:
        phase_codes = {
            str(phase["phase_id"]): str(phase["code"]) for phase in panel["phases"]
        }
        participant = str(panel.get("participant", ""))
        for point in panel["points"]:
            graph = point["graph"]
            center = rendered_centers[str(point["point_id"])]
            yield {
                "scene_id": scene["scene_id"],
                "panel_id": panel["panel_id"],
                "participant": participant,
                "series_id": point["series_id"],
                "point_id": point["point_id"],
                "observation_index": point["observation_index"],
                "printed_x_value": _csv_nullable(point.get("printed_x_value")),
                "estimated_x_value": _csv_nullable(point.get("estimated_x_value")),
                "x_confidence": point["x_confidence"],
                "x_value": graph[0],
                "y_value": graph[1],
                "phase": phase_codes[str(point["phase_id"])],
                "original_pixel_x": center[0],
                "original_pixel_y": center[1],
                "shape": point["shape"],
                "fill": point["fill"],
            }


def _csv_nullable(value: Any) -> Any:
    return "" if value is None else value


def _scene_split(scene: Mapping[str, Any]) -> str:
    splits = {
        str(family["split"]) for family in scene["families"].values()
    }
    if len(splits) != 1:
        raise AssertionError(f"Scene family split drift: {sorted(splits)}")
    return splits.pop()


def _family_keys(scene: Mapping[str, Any]) -> dict[str, str]:
    return {
        axis: str(scene["families"][axis]["key"])
        for axis in FAMILY_AXES
    }


def _split_manifests(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        selected = [case for case in cases if case["split"] == split]
        payloads[split] = {
            "manifest_version": 1,
            "split": split,
            "families": {
                axis: sorted({str(case["families"][axis]) for case in selected})
                for axis in FAMILY_AXES
            },
            "cases": [
                {
                    "scene_id": case["scene_id"],
                    "seed": case["seed"],
                    "design": case["design"],
                    "families": case["families"],
                    "files": sorted(case["files"]),
                }
                for case in selected
            ],
        }
    _assert_split_isolation(payloads)
    return payloads


def _assert_split_isolation(
    payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    for axis in FAMILY_AXES:
        family_sets = {
            split: set(payload["families"][axis])
            for split, payload in payloads.items()
        }
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = family_sets[left] & family_sets[right]
            if overlap:
                raise AssertionError(
                    f"{axis} family leakage between {left} and {right}: {sorted(overlap)}"
                )


def _validate_rendered_case(
    scene: Mapping[str, Any],
    annotation: Mapping[str, Any],
    marker_mask: Image.Image,
) -> dict[str, Any]:
    expected_points = {
        str(point["point_id"]): point
        for panel in scene["panels"]
        for point in panel["points"]
    }
    markers = _annotation_records(annotation, "markers")
    actual_markers = {
        str(marker.get("point_id", marker.get("marker_id"))): marker
        for marker in markers
    }
    if set(actual_markers) != set(expected_points):
        raise AssertionError(
            f"Marker identity drift for {scene['scene_id']}: "
            f"expected {len(expected_points)}, got {len(actual_markers)}"
        )

    for point_id, point in expected_points.items():
        expected_center = _transform_point(annotation, point["center"])
        actual_center = actual_markers[point_id]["center"]
        if not _points_close(actual_center, expected_center):
            raise AssertionError(
                f"Marker center drift for {point_id}: {actual_center} != {expected_center}"
            )
        x = round(float(expected_center[0]))
        y = round(float(expected_center[1]))
        if not 0 <= x < marker_mask.width or not 0 <= y < marker_mask.height:
            raise AssertionError(f"Marker {point_id} center is outside its mask.")
        radius = float(point["radius"])
        left = max(0, int(x - radius - 2))
        top = max(0, int(y - radius - 2))
        right = min(marker_mask.width, int(x + radius + 3))
        bottom = min(marker_mask.height, int(y + radius + 3))
        if marker_mask.crop((left, top, right, bottom)).getbbox() is None:
            raise AssertionError(f"Marker {point_id} has no pixels in its mask region.")

    expected_dividers = {
        str(divider["divider_id"]): divider
        for panel in scene["panels"]
        for divider in panel["dividers"]
    }
    actual_dividers = {
        str(divider["divider_id"]): divider
        for divider in _annotation_records(annotation, "dividers")
    }
    if set(expected_dividers) != set(actual_dividers):
        raise AssertionError("Phase-divider identity drift.")
    for divider_id, divider in expected_dividers.items():
        actual_line = actual_dividers[divider_id]["line"]
        expected_line = [
            _transform_point(annotation, divider["line"][:2]),
            _transform_point(annotation, divider["line"][2:]),
        ]
        if not all(
            _points_close(actual, expected)
            for actual, expected in zip(actual_line, expected_line, strict=True)
        ):
            raise AssertionError(
                f"Phase-divider geometry drift for {divider_id}: "
                f"{actual_line} != {expected_line}"
            )

    return {
        "markers": len(markers),
        "dividers": len(actual_dividers),
        "texts": len(_annotation_records(annotation, "texts")),
        "ticks": len(_annotation_records(annotation, "ticks")),
        "hard_negatives": len(annotation.get("hard_negatives", [])),
    }


def _annotation_records(
    annotation: Mapping[str, Any],
    key: str,
) -> list[Mapping[str, Any]]:
    records = list(annotation.get(key, []))
    for panel in annotation.get("panels", []):
        records.extend(panel.get(key, []))
    return records


def _transform_point(
    annotation: Mapping[str, Any],
    point: Sequence[float],
) -> list[float]:
    x, y = float(point[0]), float(point[1])
    transforms = annotation.get("transforms", [])
    if transforms:
        # Each emitted transform is cumulative from synthetic_clean_pixels to
        # that stage's final raster. The last record is therefore the complete
        # clean-to-final mapping and must be applied exactly once.
        matrix = transforms[-1]["forward_matrix_3x3"]
        denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if abs(denominator) < 1e-12:
            raise AssertionError("Annotation transform maps a point to infinity.")
        x, y = (
            (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
            (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
        )
        x, y = round(x, 6), round(y, 6)
    return [x, y]


def _points_close(
    actual: Sequence[float],
    expected: Sequence[float],
    *,
    tolerance: float = 1e-5,
) -> bool:
    return all(
        abs(float(actual_value) - float(expected_value)) <= tolerance
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _dataset_sanity(
    scenes: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    markers = [
        marker
        for annotation in annotations
        for marker in _annotation_records(annotation, "markers")
    ]
    dividers = [
        divider
        for annotation in annotations
        for divider in _annotation_records(annotation, "dividers")
    ]
    hard_negatives = [
        negative
        for annotation in annotations
        for negative in annotation.get("hard_negatives", [])
    ]
    texts = [
        text
        for annotation in annotations
        for text in _annotation_records(annotation, "texts")
    ]
    ticks = [
        tick
        for annotation in annotations
        for tick in _annotation_records(annotation, "ticks")
    ]
    arrows = [
        arrow
        for annotation in annotations
        for arrow in _annotation_records(annotation, "arrows")
    ]
    roles = {str(text["role"]) for text in texts}
    marker_styles = {
        (str(marker["shape"]), str(marker["fill"])) for marker in markers
    }
    line_styles = {
        str(series["line_style"])
        for scene in scenes
        for panel in scene["panels"]
        for series in panel["series"]
    }
    hard_negative_kinds = {str(item["kind"]) for item in hard_negatives}
    feature_set = {
        str(feature)
        for scene in scenes
        for feature in scene.get("layout", {}).get("features", [])
    }
    panel_counts = {len(scene["panels"]) for scene in scenes}
    session_counts = {
        int(scene["layout"]["session_count"])
        for scene in scenes
    }
    degradation_stage_counts = {
        len(scene.get("degradations", [])) for scene in scenes
    }
    x_label_visibility = {
        str(scene["presentation"]["x_label_visibility"]) for scene in scenes
    }
    y_label_visibility = {
        str(scene["presentation"]["y_label_visibility"]) for scene in scenes
    }
    legend_positions = {
        str(scene["presentation"]["legend_position"]) for scene in scenes
    }
    divider_styles = {
        str(divider["style"])
        for scene in scenes
        for panel in scene["panels"]
        for divider in panel["dividers"]
    }
    shared_axes_values = {bool(scene["layout"]["shared_axes"]) for scene in scenes}
    hidden_zero_values = {
        bool(scene["presentation"]["hide_zero_label"]) for scene in scenes
    }
    decoration_coverage = {
        key: any(bool(scene["presentation"][key]) for scene in scenes)
        for key in (
            "show_participant_names",
            "show_arrows",
            "show_brackets",
            "show_top_condition_bars",
        )
    }
    y_axis_profiles = {
        (
            float(scene["style"]["y_axis"]["minimum"]),
            float(scene["style"]["y_axis"]["maximum"]),
            float(scene["style"]["y_axis"]["tick_interval"]),
        )
        for scene in scenes
    }
    stroke_widths = {
        float(series["stroke_width"])
        for scene in scenes
        for panel in scene["panels"]
        for series in panel["series"]
    }
    marker_radii = {
        float(point["radius"])
        for scene in scenes
        for panel in scene["panels"]
        for point in panel["points"]
    }
    spacing_profiles = {
        (
            str(scene["style"]["session_spacing"]["mode"]),
            float(scene["style"]["session_spacing"]["edge_padding_fraction"]),
            float(scene["style"]["session_spacing"]["jitter_fraction"]),
            float(scene["style"]["session_spacing"]["nominal_pitch_fraction"]),
        )
        for scene in scenes
    }

    failures: list[str] = []
    expected_styles = {
        (shape, fill) for shape in MARKER_SHAPES for fill in FILL_STATES
    }
    _require_subset("marker styles", expected_styles, marker_styles, failures)
    _require_subset("line styles", set(LINE_STYLES), line_styles, failures)
    _require_subset(
        "hard negatives",
        set(HARD_NEGATIVE_KINDS),
        hard_negative_kinds,
        failures,
    )
    _require_subset("text roles", set(REQUIRED_TEXT_ROLES), roles, failures)
    _require_subset("scene features", set(SCENE_FEATURES), feature_set, failures)
    _require_subset("panel counts", {1, 6}, panel_counts, failures)
    _require_subset("session counts", {2, 100}, session_counts, failures)
    if not degradation_stage_counts or not degradation_stage_counts <= {1, 2}:
        failures.append(
            "degradation stage counts must contain only one or two stages, got "
            f"{sorted(degradation_stage_counts)}"
        )
    _require_subset(
        "x label visibility",
        {"visible", "partial", "hidden"},
        x_label_visibility,
        failures,
    )
    _require_subset(
        "legend positions",
        {"inside", "outside"},
        legend_positions,
        failures,
    )
    _require_subset(
        "divider styles",
        {"dotted"},
        divider_styles,
        failures,
    )
    if True not in shared_axes_values:
        failures.append("missing shared-axis scene")
    if True not in hidden_zero_values:
        failures.append("missing hidden-zero-label scene")
    for feature, covered in decoration_coverage.items():
        if not covered:
            failures.append(f"missing presentation decoration: {feature}")
    for label, values in (
        ("y-axis profiles", y_axis_profiles),
        ("stroke widths", stroke_widths),
        ("marker radii", marker_radii),
        ("session spacing profiles", spacing_profiles),
    ):
        if len(values) < 2:
            failures.append(f"insufficient deterministic {label}: {sorted(values, key=str)}")
    if any(not text.get("role") for text in texts):
        failures.append("one or more text annotations have no role")
    if any(tick.get("role") not in {"x_tick", "y_tick"} for tick in ticks):
        failures.append("one or more tick annotations have no x_tick/y_tick role")
    if not arrows:
        failures.append("missing annotated arrows")
    if any("line" not in arrow and "polygon" not in arrow for arrow in arrows):
        failures.append("one or more arrows have no annotated geometry")

    if failures:
        raise AssertionError("Synthetic dataset sanity failure: " + "; ".join(failures))

    return {
        "sanity_version": 1,
        "passed": True,
        "counts": {
            "scenes": len(scenes),
            "markers": len(markers),
            "dividers": len(dividers),
            "hard_negatives": len(hard_negatives),
            "csv_rows": sum(int(case["metrics"]["markers"]) for case in cases),
        },
        "coverage": {
            "designs": sorted({str(scene["design"]) for scene in scenes}),
            "panel_counts": sorted(panel_counts),
            "session_counts": sorted(session_counts),
            "marker_styles": [
                {"shape": shape, "fill": fill}
                for shape, fill in sorted(marker_styles)
            ],
            "line_styles": sorted(line_styles),
            "text_roles": sorted(roles),
            "hard_negative_kinds": sorted(hard_negative_kinds),
            "scene_features": sorted(feature_set),
            "degradation_stage_counts": sorted(degradation_stage_counts),
            "x_label_visibility": sorted(x_label_visibility),
            "y_label_visibility": sorted(y_label_visibility),
            "legend_positions": sorted(legend_positions),
            "divider_styles": sorted(divider_styles),
            "shared_axes_values": sorted(shared_axes_values),
            "hidden_zero_values": sorted(hidden_zero_values),
            "decorations": decoration_coverage,
            "y_axis_profiles": [
                {
                    "minimum": minimum,
                    "maximum": maximum,
                    "tick_interval": interval,
                }
                for minimum, maximum, interval in sorted(y_axis_profiles)
            ],
            "stroke_widths": sorted(stroke_widths),
            "marker_radii": sorted(marker_radii),
            "session_spacing_profiles": [
                {
                    "mode": mode,
                    "edge_padding_fraction": edge_padding,
                    "jitter_fraction": jitter,
                    "nominal_pitch_fraction": nominal_pitch,
                }
                for mode, edge_padding, jitter, nominal_pitch in sorted(
                    spacing_profiles
                )
            ],
            "splits": sorted({str(case["split"]) for case in cases}),
        },
    }


def _require_subset(
    label: str,
    expected: set[Any],
    actual: set[Any],
    failures: list[str],
) -> None:
    missing = expected - actual
    if missing:
        failures.append(f"missing {label}: {sorted(missing, key=str)}")


__all__ = [
    "CSV_FIELDS",
    "DatasetResult",
    "FAMILY_AXES",
    "HARD_NEGATIVE_KINDS",
    "PRESETS",
    "REQUIRED_TEXT_ROLES",
    "generate_dataset",
]
