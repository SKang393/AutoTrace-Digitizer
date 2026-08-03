# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic, local degradations for paired GraphSR-x2 training crops.

The returned HR image is the clean, geometrically augmented target.  The LR
image is derived from that target by two bounded degradation stages and an
exact x2 downsample.  Geometry is applied to the clean target before the
photometric pipeline so HR and LR stay registered.  Every random draw comes
from an operation-specific generator derived from the caller-provided seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter


REQUIRED_DEGRADATIONS: tuple[str, ...] = (
    "resize",
    "blur",
    "noise",
    "jpeg",
    "ringing",
    "paper",
    "halftone",
    "fade",
    "erosion",
    "dilation",
    "bleed",
    "skew",
    "perspective",
    "clipping",
    "jitter",
)

_STAGE_ONE_OPERATIONS = ("resize", "blur", "noise", "jpeg")
_GEOMETRY_OPERATIONS = ("skew", "perspective", "jitter")
_STAGE_TWO_OPERATIONS = tuple(
    operation for operation in REQUIRED_DEGRADATIONS if operation not in _STAGE_ONE_OPERATIONS
    and operation not in _GEOMETRY_OPERATIONS
)
_DEFAULT_PROBABILITIES: Mapping[str, float] = {
    "resize": 1.0,
    "blur": 0.90,
    "noise": 0.90,
    "jpeg": 0.85,
    "ringing": 0.40,
    "paper": 0.45,
    "halftone": 0.35,
    "fade": 0.45,
    "erosion": 0.20,
    "dilation": 0.20,
    "bleed": 0.35,
    "skew": 0.35,
    "perspective": 0.30,
    "clipping": 0.25,
    "jitter": 0.40,
}
_MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class CoordinateTransform:
    """Immutable projective transform using top-left pixel coordinates."""

    matrix: tuple[float, float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.matrix) != 9 or not all(math.isfinite(value) for value in self.matrix):
            raise ValueError("A coordinate transform must contain nine finite values")
        determinant = float(np.linalg.det(np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)))
        if abs(determinant) < 1e-12:
            raise ValueError("Coordinate transform must be invertible")

    @classmethod
    def identity(cls) -> "CoordinateTransform":
        return cls((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    @classmethod
    def scale(cls, x_scale: float, y_scale: float) -> "CoordinateTransform":
        if x_scale <= 0 or y_scale <= 0:
            raise ValueError("Scale values must be positive")
        return cls((float(x_scale), 0.0, 0.0, 0.0, float(y_scale), 0.0, 0.0, 0.0, 1.0))

    @classmethod
    def translation(cls, x: float, y: float) -> "CoordinateTransform":
        return cls((1.0, 0.0, float(x), 0.0, 1.0, float(y), 0.0, 0.0, 1.0))

    def apply(self, point: Sequence[float]) -> tuple[float, float]:
        if len(point) != 2:
            raise ValueError("A point must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Point coordinates must be finite")
        matrix = self.matrix
        denominator = matrix[6] * x + matrix[7] * y + matrix[8]
        if abs(denominator) < 1e-12:
            raise ValueError("Point maps to projective infinity")
        return (
            (matrix[0] * x + matrix[1] * y + matrix[2]) / denominator,
            (matrix[3] * x + matrix[4] * y + matrix[5]) / denominator,
        )

    def inverse(self) -> "CoordinateTransform":
        inverse = np.linalg.inv(np.asarray(self.matrix, dtype=np.float64).reshape(3, 3))
        inverse /= inverse[2, 2]
        return CoordinateTransform(tuple(float(value) for value in inverse.ravel()))

    def then(self, following: "CoordinateTransform") -> "CoordinateTransform":
        """Compose transforms so ``following`` is applied after ``self``."""

        first = np.asarray(self.matrix, dtype=np.float64).reshape(3, 3)
        second = np.asarray(following.matrix, dtype=np.float64).reshape(3, 3)
        combined = second @ first
        combined /= combined[2, 2]
        return CoordinateTransform(tuple(float(value) for value in combined.ravel()))

    def to_json(self) -> list[list[float]]:
        return [
            [float(self.matrix[row * 3 + column]) for column in range(3)]
            for row in range(3)
        ]


@dataclass(frozen=True)
class DegradationConfig:
    """Bounds and switches for the x2 degradation pipeline."""

    stage_count: int = 2
    scale: int = 2
    force_operations: tuple[str, ...] | None = None
    max_pixels: int = 16_777_216
    jpeg_quality_min: int = 42
    jpeg_quality_max: int = 92

    def __post_init__(self) -> None:
        if self.stage_count != 2:
            raise ValueError("GraphSR degradation requires exactly two stages")
        if self.scale != 2:
            raise ValueError("GraphSR degradation currently supports x2 only")
        if not 1 <= self.max_pixels <= 67_108_864:
            raise ValueError("max_pixels must be between 1 and 67,108,864")
        if not 10 <= self.jpeg_quality_min <= self.jpeg_quality_max <= 100:
            raise ValueError("JPEG quality bounds must satisfy 10 <= min <= max <= 100")
        if self.force_operations is not None:
            unknown = sorted(set(self.force_operations) - set(REQUIRED_DEGRADATIONS))
            if unknown:
                raise ValueError(f"Unknown forced degradations: {', '.join(unknown)}")
            if len(set(self.force_operations)) != len(self.force_operations):
                raise ValueError("force_operations must not contain duplicates")


@dataclass(frozen=True)
class PairedCrop:
    """An aligned clean HR target, degraded LR input, and exact geometry."""

    hr: np.ndarray
    lr: np.ndarray
    marker_centers_hr: tuple[tuple[float, float], ...]
    marker_centers_lr: tuple[tuple[float, float], ...]
    metadata: dict[str, object]
    hr_to_lr: CoordinateTransform
    lr_to_hr: CoordinateTransform
    source_to_hr: CoordinateTransform
    hr_to_source: CoordinateTransform

    def __post_init__(self) -> None:
        _validate_output_array(self.hr, "hr")
        _validate_output_array(self.lr, "lr")
        if self.hr.shape[0] != self.lr.shape[0] * 2 or self.hr.shape[1] != self.lr.shape[1] * 2:
            raise ValueError("PairedCrop must have exact x2 spatial dimensions")
        if len(self.marker_centers_hr) != len(self.marker_centers_lr):
            raise ValueError("HR and LR marker-center counts must match")
        json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        self.hr.setflags(write=False)
        self.lr.setflags(write=False)


def _validate_output_array(value: np.ndarray, name: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 uint8 RGB array")


def _as_rgb_uint8(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        converted = np.asarray(image.convert("RGB"), dtype=np.uint8)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            converted = np.repeat(image[:, :, None], 3, axis=2)
        elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
            converted = image[:, :, :3]
            if converted.shape[2] == 1:
                converted = np.repeat(converted, 3, axis=2)
        else:
            raise ValueError("Image must be HxW, HxWx1, HxWx3, or HxWx4")
        if converted.dtype != np.uint8:
            if np.issubdtype(converted.dtype, np.floating):
                if not np.isfinite(converted).all():
                    raise ValueError("Image contains non-finite values")
                peak = float(np.max(converted)) if converted.size else 0.0
                if peak <= 1.0:
                    converted = converted * 255.0
            converted = np.clip(np.rint(converted), 0, 255).astype(np.uint8)
    else:
        raise TypeError("Image must be a NumPy array or PIL image")
    return np.ascontiguousarray(converted)


def _normalize_centers(
    marker_centers: Iterable[Sequence[float]], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    normalized: list[tuple[float, float]] = []
    for index, point in enumerate(marker_centers):
        if len(point) != 2:
            raise ValueError(f"Marker center {index} must contain x and y")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"Marker center {index} must be finite")
        if not 0.0 <= x < width or not 0.0 <= y < height:
            raise ValueError(f"Marker center {index} is outside the input image")
        normalized.append((x, y))
    if len(normalized) > 100_000:
        raise ValueError("At most 100,000 marker centers are supported")
    return tuple(normalized)


def _derived_seed(seed: int, stage: int, operation: str) -> int:
    payload = f"graphsr-degradation-v1\0{seed}\0{stage}\0{operation}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & _MAX_SEED


def _rng(seed: int, stage: int, operation: str) -> np.random.Generator:
    return np.random.default_rng(_derived_seed(seed, stage, operation))


def _is_applied(config: DegradationConfig, seed: int, stage: int, operation: str) -> bool:
    if config.force_operations is not None:
        return operation in config.force_operations
    return bool(_rng(seed, stage, f"{operation}:enabled").random() < _DEFAULT_PROBABILITIES[operation])


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _solve_homography(
    source: Sequence[tuple[float, float]], destination: Sequence[tuple[float, float]]
) -> CoordinateTransform:
    rows: list[list[float]] = []
    values: list[float] = []
    for (x, y), (u, v) in zip(source, destination, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        values.append(u)
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        values.append(v)
    coefficients = np.linalg.solve(np.asarray(rows, dtype=np.float64), np.asarray(values, dtype=np.float64))
    return CoordinateTransform(tuple(float(value) for value in (*coefficients, 1.0)))


def _geometry_transform(
    width: int,
    height: int,
    seed: int,
    config: DegradationConfig,
) -> tuple[CoordinateTransform, dict[str, dict[str, object]]]:
    records: dict[str, dict[str, object]] = {}
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0

    skew_applied = _is_applied(config, seed, 2, "skew")
    skew_rng = _rng(seed, 2, "skew")
    skew_x = float(skew_rng.uniform(-0.018, 0.018)) if skew_applied else 0.0
    skew_y = float(skew_rng.uniform(-0.010, 0.010)) if skew_applied else 0.0
    skew_matrix = CoordinateTransform(
        (
            1.0,
            skew_x,
            -skew_x * center_y,
            skew_y,
            1.0,
            -skew_y * center_x,
            0.0,
            0.0,
            1.0,
        )
    )
    records["skew"] = {
        "operation": "skew",
        "applied": skew_applied,
        "execution_phase": "paired_target_preparation",
        "parameters": {"x_shear": skew_x, "y_shear": skew_y},
        "seed": _derived_seed(seed, 2, "skew"),
    }

    perspective_applied = _is_applied(config, seed, 2, "perspective")
    perspective_rng = _rng(seed, 2, "perspective")
    source_corners = (
        (0.0, 0.0),
        (float(width - 1), 0.0),
        (float(width - 1), float(height - 1)),
        (0.0, float(height - 1)),
    )
    if perspective_applied:
        magnitude = min(width, height) * float(perspective_rng.uniform(0.004, 0.018))
        offsets = perspective_rng.uniform(-magnitude, magnitude, size=(4, 2))
        destination_corners = tuple(
            (float(x + offsets[index, 0]), float(y + offsets[index, 1]))
            for index, (x, y) in enumerate(source_corners)
        )
        perspective_matrix = _solve_homography(source_corners, destination_corners)
    else:
        magnitude = 0.0
        destination_corners = source_corners
        perspective_matrix = CoordinateTransform.identity()
    records["perspective"] = {
        "operation": "perspective",
        "applied": perspective_applied,
        "execution_phase": "paired_target_preparation",
        "parameters": {
            "maximum_corner_offset_px": magnitude,
            "source_corners": [[x, y] for x, y in source_corners],
            "destination_corners": [[x, y] for x, y in destination_corners],
        },
        "seed": _derived_seed(seed, 2, "perspective"),
    }

    jitter_applied = _is_applied(config, seed, 2, "jitter")
    jitter_rng = _rng(seed, 2, "jitter")
    jitter_x = float(jitter_rng.uniform(-0.75, 0.75)) if jitter_applied else 0.0
    jitter_y = float(jitter_rng.uniform(-0.75, 0.75)) if jitter_applied else 0.0
    jitter_matrix = CoordinateTransform.translation(jitter_x, jitter_y)
    records["jitter"] = {
        "operation": "jitter",
        "applied": jitter_applied,
        "execution_phase": "paired_target_preparation",
        "parameters": {"x_px": jitter_x, "y_px": jitter_y},
        "seed": _derived_seed(seed, 2, "jitter"),
    }

    combined = skew_matrix.then(perspective_matrix).then(jitter_matrix)
    return combined, records


def _warp(image: Image.Image, forward: CoordinateTransform) -> Image.Image:
    inverse = forward.inverse().matrix
    coefficients = tuple(float(value) for value in inverse[:8])
    return image.transform(
        image.size,
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def _apply_resize(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    factor = float(rng.uniform(0.72, 1.18))
    width = max(8, int(round(image.width * factor)))
    height = max(8, int(round(image.height * factor)))
    downsample = factor < 1.0
    first_resampling = Image.Resampling.LANCZOS if downsample else Image.Resampling.BICUBIC
    resized = image.resize((width, height), first_resampling)
    return resized, {
        "factor": factor,
        "input_size": [image.width, image.height],
        "output_size": [width, height],
        "resampling": "lanczos" if downsample else "bicubic",
    }


def _apply_blur(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    radius = float(rng.uniform(0.18, 1.15))
    return image.filter(ImageFilter.GaussianBlur(radius)), {"kind": "gaussian", "radius": radius}


def _apply_noise(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    array = np.asarray(image, dtype=np.float32)
    sigma = float(rng.uniform(0.8, 6.0))
    monochrome = bool(rng.random() < 0.65)
    shape = (*array.shape[:2], 1) if monochrome else array.shape
    noise = rng.normal(0.0, sigma, shape).astype(np.float32)
    degraded = np.clip(np.rint(array + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, "RGB"), {"sigma": sigma, "monochrome": monochrome}


def _apply_jpeg(
    image: Image.Image, rng: np.random.Generator, config: DegradationConfig
) -> tuple[Image.Image, dict[str, object]]:
    quality = int(rng.integers(config.jpeg_quality_min, config.jpeg_quality_max + 1))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=False)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        decoded = encoded.convert("RGB").copy()
    return decoded, {"quality": quality, "chroma_subsampling": "4:4:4"}


def _apply_ringing(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    amount = int(rng.integers(65, 151))
    radius = float(rng.uniform(0.7, 1.35))
    threshold = int(rng.integers(1, 5))
    result = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=amount, threshold=threshold))
    return result, {"method": "unsharp_overshoot", "radius": radius, "percent": amount, "threshold": threshold}


def _apply_paper(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    cell = int(rng.integers(12, 29))
    grid_width = max(2, math.ceil(width / cell) + 1)
    grid_height = max(2, math.ceil(height / cell) + 1)
    low_frequency = rng.normal(0.0, 1.0, (grid_height, grid_width)).astype(np.float32)
    texture = np.asarray(
        Image.fromarray(low_frequency, mode="F").resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ).copy()
    texture -= float(texture.mean())
    texture /= max(float(texture.std()), 1e-6)
    amplitude = float(rng.uniform(1.0, 5.0))
    tinted = np.clip(np.rint(array + texture[:, :, None] * amplitude), 0, 255).astype(np.uint8)
    return Image.fromarray(tinted, "RGB"), {"cell_size_px": cell, "amplitude": amplitude}


def _apply_halftone(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    array = np.asarray(image, dtype=np.float32)
    height, width = array.shape[:2]
    period = float(rng.uniform(3.2, 6.5))
    angle = float(rng.uniform(-math.pi / 6.0, math.pi / 6.0))
    amplitude = float(rng.uniform(2.0, 8.0))
    yy, xx = np.mgrid[:height, :width]
    coordinate = xx * math.cos(angle) + yy * math.sin(angle)
    pattern = np.sin(coordinate * math.tau / period).astype(np.float32)
    ink_weight = np.clip((255.0 - np.mean(array, axis=2)) / 255.0, 0.0, 1.0)
    degraded = np.clip(np.rint(array + pattern[:, :, None] * amplitude * ink_weight[:, :, None]), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, "RGB"), {"period_px": period, "angle_radians": angle, "amplitude": amplitude}


def _apply_fade(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    amount = float(rng.uniform(0.025, 0.16))
    array = np.asarray(image, dtype=np.float32)
    faded = np.clip(np.rint(array * (1.0 - amount) + 255.0 * amount), 0, 255).astype(np.uint8)
    return Image.fromarray(faded, "RGB"), {"white_blend": amount}


def _apply_morphology(
    image: Image.Image, operation: str, rng: np.random.Generator
) -> tuple[Image.Image, dict[str, object]]:
    size = int(rng.choice((3, 3, 3, 5)))
    # Dark pixels represent printed ink.  MaxFilter erodes dark ink and
    # MinFilter dilates it.
    image_filter = ImageFilter.MaxFilter(size) if operation == "erosion" else ImageFilter.MinFilter(size)
    return image.filter(image_filter), {"kernel_size": size, "foreground": "dark_ink"}


def _apply_bleed(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    array = np.asarray(image, dtype=np.float32)
    offset_x = int(rng.integers(-2, 3))
    offset_y = int(rng.integers(-2, 3))
    opacity = float(rng.uniform(0.025, 0.09))
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.6, 1.4)))), dtype=np.float32)
    shifted = np.roll(blurred, shift=(offset_y, offset_x), axis=(0, 1))
    bleed_ink = 255.0 - shifted
    degraded = np.clip(np.rint(array - bleed_ink * opacity), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, "RGB"), {"offset_px": [offset_x, offset_y], "opacity": opacity}


def _apply_clipping(image: Image.Image, rng: np.random.Generator) -> tuple[Image.Image, dict[str, object]]:
    width, height = image.size
    side = str(rng.choice(("left", "right", "top", "bottom")))
    maximum = max(1, int(round((width if side in ("left", "right") else height) * 0.035)))
    amount = int(rng.integers(1, maximum + 1))
    array = np.asarray(image, dtype=np.uint8).copy()
    if side == "left":
        array[:, :amount] = 255
    elif side == "right":
        array[:, width - amount :] = 255
    elif side == "top":
        array[:amount, :] = 255
    else:
        array[height - amount :, :] = 255
    return Image.fromarray(array, "RGB"), {
        "side": side,
        "amount_px": amount,
        "coordinate_effect": "content_loss_only",
        "reversible": False,
    }


def _operation_record(
    operation: str,
    applied: bool,
    seed: int,
    parameters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "operation": operation,
        "applied": bool(applied),
        "seed": int(seed),
        "parameters": dict(parameters or {}),
    }


def _apply_stage_one(
    image: Image.Image, seed: int, config: DegradationConfig
) -> tuple[Image.Image, list[dict[str, object]]]:
    result = image
    records: list[dict[str, object]] = []
    for operation in _STAGE_ONE_OPERATIONS:
        operation_seed = _derived_seed(seed, 1, operation)
        applied = _is_applied(config, seed, 1, operation)
        parameters: dict[str, object] = {}
        if applied:
            generator = _rng(seed, 1, operation)
            if operation == "resize":
                result, parameters = _apply_resize(result, generator)
            elif operation == "blur":
                result, parameters = _apply_blur(result, generator)
            elif operation == "noise":
                result, parameters = _apply_noise(result, generator)
            elif operation == "jpeg":
                result, parameters = _apply_jpeg(result, generator, config)
        records.append(_operation_record(operation, applied, operation_seed, parameters))
    return result, records


def _apply_stage_two(
    image: Image.Image,
    target_size: tuple[int, int],
    seed: int,
    config: DegradationConfig,
) -> tuple[Image.Image, list[dict[str, object]]]:
    result = image
    records: list[dict[str, object]] = []
    for operation in _STAGE_TWO_OPERATIONS:
        operation_seed = _derived_seed(seed, 2, operation)
        applied = _is_applied(config, seed, 2, operation)
        parameters: dict[str, object] = {}
        if applied:
            generator = _rng(seed, 2, operation)
            if operation == "ringing":
                result, parameters = _apply_ringing(result, generator)
            elif operation == "paper":
                result, parameters = _apply_paper(result, generator)
            elif operation == "halftone":
                result, parameters = _apply_halftone(result, generator)
            elif operation == "fade":
                result, parameters = _apply_fade(result, generator)
            elif operation in ("erosion", "dilation"):
                result, parameters = _apply_morphology(result, operation, generator)
            elif operation == "bleed":
                result, parameters = _apply_bleed(result, generator)
            elif operation == "clipping":
                result, parameters = _apply_clipping(result, generator)
        records.append(_operation_record(operation, applied, operation_seed, parameters))

    input_size = [result.width, result.height]
    result = result.resize(target_size, Image.Resampling.LANCZOS)
    records.append(
        {
            "operation": "final_downsample",
            "applied": True,
            "seed": None,
            "parameters": {
                "input_size": input_size,
                "output_size": [target_size[0], target_size[1]],
                "scale": config.scale,
                "resampling": "lanczos",
            },
        }
    )
    return result, records


def build_paired_crop(
    hr: np.ndarray | Image.Image,
    marker_centers: Iterable[Sequence[float]],
    seed: int,
    config: DegradationConfig | None = None,
) -> PairedCrop:
    """Create one deterministic, exactly aligned x2 HR/LR training pair.

    ``marker_centers`` are expressed in the input crop's top-left pixel space.
    The returned marker centers use the returned HR and LR pixel spaces.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    effective_config = config or DegradationConfig()
    image_array = _as_rgb_uint8(hr)
    height, width = image_array.shape[:2]
    if width < 16 or height < 16:
        raise ValueError("HR crops must be at least 16 by 16 pixels")
    if width % effective_config.scale or height % effective_config.scale:
        raise ValueError("HR crop dimensions must be divisible by the x2 scale")
    if width * height > effective_config.max_pixels:
        raise ValueError("HR crop exceeds the configured pixel bound")
    centers_source = _normalize_centers(marker_centers, width, height)

    source_image = Image.fromarray(image_array, "RGB")
    source_to_hr, geometry_records = _geometry_transform(width, height, seed, effective_config)
    clean_hr_image = _warp(source_image, source_to_hr)
    clean_hr = np.ascontiguousarray(np.asarray(clean_hr_image, dtype=np.uint8))
    centers_hr = tuple(source_to_hr.apply(point) for point in centers_source)

    stage_one_image, stage_one_records = _apply_stage_one(clean_hr_image, seed, effective_config)
    target_size = (width // effective_config.scale, height // effective_config.scale)
    low_resolution_image, stage_two_records = _apply_stage_two(
        stage_one_image,
        target_size,
        seed,
        effective_config,
    )
    low_resolution = np.ascontiguousarray(np.asarray(low_resolution_image, dtype=np.uint8))

    hr_to_lr = CoordinateTransform.scale(
        low_resolution.shape[1] / clean_hr.shape[1],
        low_resolution.shape[0] / clean_hr.shape[0],
    )
    lr_to_hr = hr_to_lr.inverse()
    centers_lr = tuple(hr_to_lr.apply(point) for point in centers_hr)
    hr_to_source = source_to_hr.inverse()

    metadata: dict[str, object] = {
        "contract": "graphsr-paired-degradation-v1",
        "seed": seed,
        "scale": effective_config.scale,
        "coordinate_space": "top-left-pixel-center",
        "input": {
            "width": width,
            "height": height,
            "mode": "RGB",
            "sha256": _array_hash(image_array),
        },
        "output": {
            "hr_width": int(clean_hr.shape[1]),
            "hr_height": int(clean_hr.shape[0]),
            "lr_width": int(low_resolution.shape[1]),
            "lr_height": int(low_resolution.shape[0]),
            "hr_sha256": _array_hash(clean_hr),
            "lr_sha256": _array_hash(low_resolution),
        },
        "geometry": {
            "source_to_hr": source_to_hr.to_json(),
            "hr_to_source": hr_to_source.to_json(),
            "hr_to_lr": hr_to_lr.to_json(),
            "lr_to_hr": lr_to_hr.to_json(),
            "reversible_for_retained_coordinates": True,
            "clipping_is_content_loss": True,
        },
        "geometry_preparation": {
            "execution_order": 0,
            "applied_before": "stage_1",
            "purpose": "apply one recorded warp to both the clean target and degraded input so paired pixels remain aligned",
            "operations": [dict(geometry_records[name]) for name in _GEOMETRY_OPERATIONS],
        },
        "marker_centers": {
            "source": [[x, y] for x, y in centers_source],
            "hr": [[x, y] for x, y in centers_hr],
            "lr": [[x, y] for x, y in centers_lr],
        },
        "stages": [
            {"index": 1, "operations": stage_one_records},
            {"index": 2, "operations": stage_two_records},
        ],
    }
    json.dumps(metadata, sort_keys=True, allow_nan=False)

    return PairedCrop(
        hr=clean_hr,
        lr=low_resolution,
        marker_centers_hr=centers_hr,
        marker_centers_lr=centers_lr,
        metadata=metadata,
        hr_to_lr=hr_to_lr,
        lr_to_hr=lr_to_hr,
        source_to_hr=source_to_hr,
        hr_to_source=hr_to_source,
    )


__all__ = [
    "CoordinateTransform",
    "DegradationConfig",
    "PairedCrop",
    "REQUIRED_DEGRADATIONS",
    "build_paired_crop",
]
