# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Run the single preregistered, non-approval PP-OCRv5 detector diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from ml.ocr.official_bakeoff import structure_consensus_evaluate as detector_contract


PROFILE = "graphreader-ocr-combined-v5-detector-diagnostic-v1"
PROTOCOL_SCHEMA = "graphreader.ocr-combined-v5-detector-diagnostic-protocol.v1"
REPORT_SCHEMA = "graphreader.ocr-combined-v5-detector-diagnostic.v1"
DETECTOR_SHA256 = "d4aa24d408cd70b8b9f66cc758e20f397fc31a9c69d8477cf8887fc53bd5fceb"
EXPECTED_CASES = 72
TEXT_CASES = 48
EXCLUSION_CASES = 24
WIDTH = 384
HEIGHT = 192


class DiagnosticError(RuntimeError):
    """Raised when the frozen diagnostic contract cannot be honored."""


@dataclass(frozen=True)
class RenderedCase:
    case_id: str
    kind: str
    source_png: bytes
    source_sha256: str
    detector_bgr: bytes
    detector_bgr_sha256: str
    truth_bbox: tuple[float, float, float, float] | None
    structure_family: str
    degradation_family: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _protocol_path() -> Path:
    return Path(__file__).resolve().with_name("DETECTOR_DIAGNOSTIC_PROTOCOL.json")


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    return _hash_bytes(path.read_bytes())


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DiagnosticError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    _require(isinstance(value, dict), "Diagnostic protocol must be a JSON object.")
    return value


def validate_protocol(protocol_path: Path, workflow_path: Path | None = None) -> dict[str, Any]:
    workflow_path = workflow_path or Path(__file__).resolve()
    protocol = _load_json(protocol_path)
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "Diagnostic protocol schema changed.")
    _require(protocol.get("profile") == PROFILE, "Diagnostic profile changed.")
    _require(protocol.get("status") == "frozen_before_diagnostic_inference", "Diagnostic was not frozen before inference.")
    _require(protocol.get("purpose") == "non_approval_detector_defect_characterization", "Diagnostic purpose changed.")
    _require(protocol.get("production_approval") is False, "A diagnostic cannot grant production approval.")
    _require(protocol.get("release_eligible") is False, "A diagnostic cannot be release eligible.")
    _require(protocol.get("private_data") is False and protocol.get("chandler_used") is False, "Private or Chandler data is prohibited.")
    _require(protocol.get("detector_onnx_sha256") == DETECTOR_SHA256, "Detector selection changed.")
    _require(protocol.get("execution_workflow_sha256") == _hash_file(workflow_path), "Frozen diagnostic workflow changed.")
    _require(protocol.get("case_count") == EXPECTED_CASES, "Diagnostic case count changed.")
    _require(protocol.get("renderer") == {
        "family": "offset-lattice-graph-diagnostic-v1",
        "seed": 20260831,
        "width": WIDTH,
        "height": HEIGHT,
        "text_cases": TEXT_CASES,
        "exclusion_cases": EXCLUSION_CASES,
    }, "Diagnostic renderer contract changed.")
    _require(protocol.get("experiment_budget") == {
        "fixture_freezes": 1,
        "detector_diagnostic_runs": 1,
        "post_inference_threshold_changes": 0,
        "promotion_decisions": 0,
    }, "Diagnostic experiment budget changed.")
    exposed = protocol.get("prior_exposed_splits_forbidden")
    _require(isinstance(exposed, list) and len(exposed) == 2, "Exposed split denial changed.")
    for item in exposed:
        _require(isinstance(item, dict), "Exposed split denial is invalid.")
        for key in ("split_sha256", "fixture_archive_sha256"):
            value = item.get(key)
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value.lower()),
                f"Exposed {key} is invalid.",
            )
    sources = protocol.get("reviewed_source_sha256")
    _require(isinstance(sources, dict) and sources, "Reviewed source inventory is missing.")
    root = _repo_root()
    for relative, expected in sources.items():
        source = (root / str(relative)).resolve()
        _require(source.is_relative_to(root) and source.is_file(), f"Reviewed source is missing: {relative}")
        _require(_hash_file(source) == expected, f"Reviewed source changed: {relative}")
    return protocol


def _draw_structure(draw: Any, index: int) -> tuple[str, list[dict[str, int]]]:
    family = ("offset_lattice", "staggered_phase", "curved_connectors", "boxed_legend")[index % 4]
    masks: list[dict[str, int]] = []

    def mask(kind: str, left: int, top: int, right: int, bottom: int) -> None:
        masks.append({"kind": kind, "left": left, "top": top, "right": right, "bottom": bottom})

    axis_x = 62 + (index % 3)
    axis_y = 158 - (index % 4)
    draw.line((axis_x, 38, axis_x, axis_y), fill=(32, 35, 38), width=2)
    draw.line((axis_x, axis_y, 356, axis_y), fill=(32, 35, 38), width=2)
    mask("y_axis", axis_x - 3, 35, axis_x + 4, axis_y + 4)
    mask("x_axis", axis_x - 3, axis_y - 3, 360, axis_y + 4)
    for tick in range(6):
        x = axis_x + 31 + (tick * 43)
        y = axis_y - 25 - ((tick * 17 + index * 5) % 77)
        draw.line((x, axis_y - 4, x, axis_y + 4), fill=(35, 38, 41), width=1)
        draw.line((axis_x - 4, y, axis_x + 4, y), fill=(35, 38, 41), width=1)
        mask("x_tick", x - 2, axis_y - 6, x + 3, axis_y + 7)
        mask("y_tick", axis_x - 6, y - 2, axis_x + 7, y + 3)
    if family == "staggered_phase":
        x = 207 + (index % 5)
        draw.line((x, 29, x, axis_y), fill=(42, 42, 42), width=2)
        mask("phase_divider", x - 3, 26, x + 4, axis_y + 3)
    if family in {"curved_connectors", "offset_lattice"}:
        points = [(104, 113), (145, 87), (189, 105), (234, 69), (280, 91)]
        draw.line(points, fill=(34, 34, 34), width=2, joint="curve")
        for point_index, (x, y) in enumerate(points):
            if (point_index + index) % 2:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(24, 24, 24), fill=(252, 252, 250), width=2)
            else:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(24, 24, 24))
    if family == "boxed_legend":
        draw.rounded_rectangle((252, 36, 354, 79), radius=3, outline=(36, 36, 36), width=2)
        draw.ellipse((264, 51, 273, 60), fill=(28, 28, 28))
        draw.line((281, 56, 338, 56), fill=(38, 38, 38), width=2)
    return family, masks


def _degrade(image: Any, index: int) -> tuple[Any, str]:
    from io import BytesIO
    from PIL import Image, ImageEnhance, ImageFilter

    family = ("warm_paper", "anisotropic_restore", "low_contrast_scan", "jpeg_chroma")[index % 4]
    if family == "warm_paper":
        return image, family
    if family == "anisotropic_restore":
        reduced = image.resize((301, 171), Image.Resampling.BICUBIC)
        return reduced.resize(image.size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(0.25)), family
    if family == "low_contrast_scan":
        return ImageEnhance.Contrast(image.filter(ImageFilter.GaussianBlur(0.35))).enhance(0.82), family
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=79, subsampling=2, optimize=False, progressive=False)
    stream.seek(0)
    with Image.open(stream) as loaded:
        return loaded.convert("RGB"), family


def render_case(index: int, font_path: Path) -> RenderedCase:
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont

    _require(0 <= index < EXPECTED_CASES, "Diagnostic case index is out of range.")
    _require(font_path.is_file(), "Diagnostic font is missing.")
    kind = "text" if index < TEXT_CASES else "exclusion"
    image = Image.new("RGB", (WIDTH, HEIGHT), (252, 252, 250))
    draw = ImageDraw.Draw(image)
    structure_family, masks = _draw_structure(draw, index)
    truth_bbox: tuple[float, float, float, float] | None = None
    if kind == "text":
        values = ("0", "17", "-8", "3.5", "42%", "100", "-0.7", "9.25")
        text = values[(index * 5 + 3) % len(values)]
        font = ImageFont.truetype(str(font_path), 18 + ((index * 7) % 11))
        anchor = ((91, 15), (294, 91), (16, 89), (157, 15), (314, 125), (94, 130))[index % 6]
        draw.text(anchor, text, font=font, fill=(20, 22, 24), stroke_width=0)
        box = draw.textbbox(anchor, text, font=font, stroke_width=0)
        truth_bbox = tuple(float(value) for value in box)
    image, degradation_family = _degrade(image, index)
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=9)
    png = stream.getvalue()
    exact = Image.open(BytesIO(png)).convert("RGB")
    rgb = np.asarray(exact, dtype=np.uint8)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    for rectangle in masks:
        bgr[rectangle["top"]:rectangle["bottom"], rectangle["left"]:rectangle["right"], :] = 255
    detector_bgr = bgr.tobytes(order="C")
    return RenderedCase(
        case_id=f"diagnostic-{kind}-{index:03d}",
        kind=kind,
        source_png=png,
        source_sha256=_hash_bytes(png),
        detector_bgr=detector_bgr,
        detector_bgr_sha256=_hash_bytes(detector_bgr),
        truth_bbox=truth_bbox,
        structure_family=structure_family,
        degradation_family=degradation_family,
    )


def _iou(region: Any, truth: tuple[float, float, float, float]) -> float:
    left, top, right, bottom = truth
    intersection_width = max(0.0, min(region.bounds.right, right) - max(region.bounds.left, left))
    intersection_height = max(0.0, min(region.bounds.bottom, bottom) - max(region.bounds.top, top))
    intersection = intersection_width * intersection_height
    union = region.bounds.width * region.bounds.height + (right - left) * (bottom - top) - intersection
    return 0.0 if union <= 0 else intersection / union


def diagnose_case(session: Any, case: RenderedCase) -> dict[str, Any]:
    tensor = detector_contract.detector_tensor(case.detector_bgr, WIDTH, HEIGHT)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _require(len(inputs) == 1 and len(outputs) == 1, "Detector ONNX contract changed.")
    started = perf_counter()
    raw = np.asarray(session.run([outputs[0].name], {inputs[0].name: tensor})[0], dtype=np.float32)
    duration_ms = (perf_counter() - started) * 1000.0
    expected_shape = (1, 1, int(tensor.shape[2]), int(tensor.shape[3]))
    shape_valid = tuple(int(value) for value in raw.shape) == expected_shape
    finite = bool(np.isfinite(raw).all())
    finite_values = raw[np.isfinite(raw)]
    minimum = float(finite_values.min()) if finite_values.size else None
    maximum = float(finite_values.max()) if finite_values.size else None
    strict_probability = (
        finite
        and minimum is not None
        and maximum is not None
        and minimum >= 0.0
        and maximum <= 1.0
    )
    regions: Sequence[Any] = ()
    final_regions: Sequence[Any] = ()
    if shape_valid and finite:
        clipped = np.clip(raw, np.float32(0.0), np.float32(1.0))
        regions = detector_contract.db_model_regions(clipped, WIDTH, HEIGHT)
        candidates = detector_contract.connected_component_candidates(case.detector_bgr, WIDTH, HEIGHT)
        _, final_regions = detector_contract.compose_consensus(regions, candidates)
    truth_matches = [] if case.truth_bbox is None else [
        _iou(region, case.truth_bbox) for region in final_regions
    ]
    best_iou = max(truth_matches, default=0.0)
    matched = int(case.truth_bbox is not None and best_iou >= 0.5)
    expected = 1 if case.truth_bbox is not None else 0
    false_regions = max(0, len(final_regions) - matched)
    return {
        "case_id": case.case_id,
        "kind": case.kind,
        "source_sha256": case.source_sha256,
        "detector_bgr_sha256": case.detector_bgr_sha256,
        "structure_family": case.structure_family,
        "degradation_family": case.degradation_family,
        "truth_bbox": list(case.truth_bbox) if case.truth_bbox is not None else None,
        "detector_input_tensor": {"sha256": _hash_bytes(tensor.tobytes(order="C")), "dtype": "float32", "shape": list(tensor.shape)},
        "detector_output_tensor": {"sha256": _hash_bytes(raw.tobytes(order="C")), "dtype": "float32", "shape": list(raw.shape)},
        "raw_minimum": minimum,
        "raw_maximum": maximum,
        "strict_probability": strict_probability,
        "shape_valid": shape_valid,
        "finite": finite,
        "diagnostic_clamp_applied_for_region_analysis": shape_valid and finite and not strict_probability,
        "raw_db_region_count": len(regions),
        "final_region_count": len(final_regions),
        "expected_region_count": expected,
        "matched_region_count": matched,
        "false_region_count": false_regions,
        "best_truth_iou": best_iou,
        "duration_ms": duration_ms,
    }


def build_report(protocol: dict[str, Any], model_path: Path, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    _require(len(records) == EXPECTED_CASES, "Diagnostic record count changed.")
    strict_violations = sum(not bool(record["strict_probability"]) for record in records)
    nonfinite = sum(not bool(record["finite"]) for record in records)
    shape_failures = sum(not bool(record["shape_valid"]) for record in records)
    exact = sum(
        int(record["matched_region_count"]) == int(record["expected_region_count"])
        and int(record["false_region_count"]) == 0
        for record in records
    )
    text_exact = sum(
        int(record["matched_region_count"]) == 1 and int(record["false_region_count"]) == 0
        for record in records if record["kind"] == "text"
    )
    exclusion_exact = sum(
        int(record["final_region_count"]) == 0
        for record in records if record["kind"] == "exclusion"
    )
    finite_minimums = [float(record["raw_minimum"]) for record in records if record["raw_minimum"] is not None]
    finite_maximums = [float(record["raw_maximum"]) for record in records if record["raw_maximum"] is not None]
    minimum = min(finite_minimums) if finite_minimums else None
    maximum = max(finite_maximums) if finite_maximums else None
    return {
        "schema": REPORT_SCHEMA,
        "profile": PROFILE,
        "status": "diagnostic_complete",
        "purpose": "non_approval_detector_defect_characterization",
        "production_approval": False,
        "release_eligible": False,
        "private_data": False,
        "chandler_used": False,
        "provider": "CPUExecutionProvider",
        "protocol_sha256": _hash_file(_protocol_path()),
        "workflow_sha256": _hash_file(Path(__file__)),
        "detector_onnx_sha256": _hash_file(model_path),
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": sys.version,
            "python_executable_sha256": _hash_file(Path(sys.executable)),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "onnxruntime_version": __import__("onnxruntime").__version__,
            "opencv_version": __import__("cv2").__version__,
            "pillow_version": __import__("PIL").__version__,
        },
        "metrics": {
            "case_count": len(records),
            "strict_probability_violation_count": strict_violations,
            "nonfinite_output_count": nonfinite,
            "shape_failure_count": shape_failures,
            "raw_minimum": minimum,
            "raw_maximum": maximum,
            "maximum_underflow": None if minimum is None else max(0.0, -minimum),
            "maximum_overflow": None if maximum is None else max(0.0, maximum - 1.0),
            "composition_exact_rate": exact / len(records),
            "text_detection_exact_rate": text_exact / TEXT_CASES,
            "exclusion_exact_rate": exclusion_exact / EXCLUSION_CASES,
            "false_region_count": sum(int(record["false_region_count"]) for record in records),
            "total_duration_ms": sum(float(record["duration_ms"]) for record in records),
        },
        "interpretation": {
            "may_inform_future_preregistration": True,
            "may_change_production_thresholds": False,
            "may_create_manifest": False,
            "may_promote_model_store": False,
            "future_gate_must_use_new_fixture_families_and_seed": True,
        },
        "records": list(records),
    }


def run_diagnostic(protocol_path: Path, model_path: Path, output_root: Path, font_path: Path) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path)
    _require(not output_root.exists(), "Refusing to overwrite the one-run diagnostic output.")
    _require(model_path.is_file(), "Exact detector ONNX is missing.")
    _require(_hash_file(model_path) == DETECTOR_SHA256, "Detector ONNX checksum changed.")
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    records: list[dict[str, Any]] = []
    assets: list[tuple[str, bytes]] = []
    for index in range(EXPECTED_CASES):
        case = render_case(index, font_path)
        assets.append((case.case_id, case.source_png))
        records.append(diagnose_case(session, case))
    report = build_report(protocol, model_path, records)
    output_root.mkdir(parents=True, exist_ok=False)
    asset_root = output_root / "assets"
    asset_root.mkdir()
    for case_id, png in assets:
        (asset_root / f"{case_id}.png").write_bytes(png)
    (output_root / "report.json").write_bytes(_canonical(report))
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=_protocol_path())
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=_repo_root() / "src" / "GraphReader.App" / "Assets" / "Fonts" / "NotoSans-Regular.ttf",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_diagnostic(args.protocol, args.model, args.output_root, args.font)
        print(json.dumps({"status": report["status"], **report["metrics"]}, sort_keys=True))
    except (DiagnosticError, OSError, ValueError) as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
