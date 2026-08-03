# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Marker-aware paired crop construction for GraphSR training and tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
from PIL import Image

from .degradation import (
    CoordinateTransform,
    DegradationConfig,
    PairedCrop,
    build_paired_crop,
)


DATASET_CONTRACT = "graphsr-training-pair-v1"
_MAX_PAIR_COUNT = 10_000
_MAX_CROP_SIDE = 4_096
_MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class PairedTrainingSample:
    """One immutable training sample with source-to-model provenance."""

    sample_id: str
    seed: int
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
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if self.hr.dtype != np.uint8 or self.lr.dtype != np.uint8:
            raise ValueError("Training images must be uint8")
        if self.hr.shape[0] != self.lr.shape[0] * 2 or self.hr.shape[1] != self.lr.shape[1] * 2:
            raise ValueError("Training pairs must have exact x2 dimensions")
        json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        self.hr.setflags(write=False)
        self.lr.setflags(write=False)


class GraphSrPairDataset(Sequence[PairedTrainingSample]):
    """Minimal framework-neutral sequence wrapper used by training adapters."""

    def __init__(self, samples: Iterable[PairedTrainingSample]) -> None:
        self._samples = tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int | slice) -> PairedTrainingSample | tuple[PairedTrainingSample, ...]:
        return self._samples[index]


def _as_rgb_uint8(image: np.ndarray | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array or PIL image")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] in (1, 3, 4):
        image = image[:, :, :3]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
    else:
        raise ValueError("image must be HxW, HxWx1, HxWx3, or HxWx4")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if not np.isfinite(image).all():
                raise ValueError("image contains non-finite values")
            if image.size and float(np.max(image)) <= 1.0:
                image = image * 255.0
        image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _point_from_value(value: object, label: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if "center" in value:
            return _point_from_value(value["center"], f"{label}.center")
        if "x" in value and "y" in value:
            value = (value["x"], value["y"])
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain x and y")
    x, y = float(value[0]), float(value[1])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"{label} coordinates must be finite")
    return x, y


def extract_marker_centers(annotation: Mapping[str, object]) -> tuple[tuple[float, float], ...]:
    """Extract stable marker centers from the synthetic-renderer annotation.

    The renderer currently provides a flat top-level ``markers`` collection.
    Panel-local markers are also accepted for fixture and forward compatibility.
    Duplicate marker IDs or coordinates are returned once.
    """

    if not isinstance(annotation, Mapping):
        raise TypeError("annotation must be a mapping")
    candidates: list[Mapping[str, object]] = []
    top_level = annotation.get("markers", ())
    if isinstance(top_level, Sequence) and not isinstance(top_level, (str, bytes)):
        candidates.extend(item for item in top_level if isinstance(item, Mapping))
    panels = annotation.get("panels", ())
    if isinstance(panels, Sequence) and not isinstance(panels, (str, bytes)):
        for panel in panels:
            if not isinstance(panel, Mapping):
                continue
            panel_markers = panel.get("markers", ())
            if isinstance(panel_markers, Sequence) and not isinstance(panel_markers, (str, bytes)):
                candidates.extend(item for item in panel_markers if isinstance(item, Mapping))

    result: list[tuple[float, float]] = []
    seen_keys: set[tuple[str, object]] = set()
    for index, marker in enumerate(candidates):
        if "center" not in marker:
            continue
        center = _point_from_value(marker["center"], f"marker[{index}]")
        marker_id = marker.get("marker_id", marker.get("point_id"))
        key: tuple[str, object]
        if marker_id is not None:
            key = ("id", str(marker_id))
        else:
            key = ("point", (round(center[0], 9), round(center[1], 9)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        result.append(center)
    return tuple(result)


def _normalize_centers_input(
    annotation_or_centers: Mapping[str, object] | Iterable[Sequence[float]],
    width: int,
    height: int,
) -> tuple[tuple[float, float], ...]:
    if isinstance(annotation_or_centers, Mapping):
        centers = extract_marker_centers(annotation_or_centers)
    else:
        centers = tuple(
            _point_from_value(value, f"marker_centers[{index}]")
            for index, value in enumerate(annotation_or_centers)
        )
    for index, (x, y) in enumerate(centers):
        if not 0.0 <= x < width or not 0.0 <= y < height:
            raise ValueError(f"Marker center {index} is outside the source image")
    return centers


def _crop_size(value: int | Sequence[int], image_width: int, image_height: int) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError("crop_size must be an integer or (width, height)")
    if isinstance(value, int):
        width = height = value
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
    else:
        raise TypeError("crop_size must be an integer or (width, height)")
    if width < 16 or height < 16 or width > _MAX_CROP_SIDE or height > _MAX_CROP_SIDE:
        raise ValueError(f"crop dimensions must be between 16 and {_MAX_CROP_SIDE}")
    if width % 2 or height % 2:
        raise ValueError("crop dimensions must be even for exact x2 pairs")
    if width > image_width or height > image_height:
        raise ValueError("crop dimensions cannot exceed the source image")
    return width, height


def _derived_pair_seed(seed: int, index: int) -> int:
    payload = f"graphsr-dataset-v1\0{seed}\0{index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & _MAX_SEED


def _crop_rng(seed: int) -> np.random.Generator:
    payload = f"graphsr-crop-selection-v1\0{seed}".encode("ascii")
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & _MAX_SEED
    return np.random.default_rng(derived)


def _select_origins(
    image_width: int,
    image_height: int,
    crop_width: int,
    crop_height: int,
    centers: tuple[tuple[float, float], ...],
    seed: int,
    count: int,
) -> tuple[tuple[int, int], ...]:
    if image_width == crop_width and image_height == crop_height:
        return tuple((0, 0) for _ in range(count))
    generator = _crop_rng(seed)
    origins: list[tuple[int, int]] = []
    marker_order = generator.permutation(len(centers)).tolist() if centers else []
    for index in range(count):
        use_marker = bool(centers) and (index % 3 != 2)
        if use_marker:
            center = centers[marker_order[index % len(marker_order)]]
            jitter_x = float(generator.uniform(-crop_width * 0.12, crop_width * 0.12))
            jitter_y = float(generator.uniform(-crop_height * 0.12, crop_height * 0.12))
            left = int(round(center[0] - crop_width / 2.0 + jitter_x))
            top = int(round(center[1] - crop_height / 2.0 + jitter_y))
            left = min(max(left, 0), image_width - crop_width)
            top = min(max(top, 0), image_height - crop_height)
        else:
            left = int(generator.integers(0, image_width - crop_width + 1))
            top = int(generator.integers(0, image_height - crop_height + 1))
        origins.append((left, top))
    return tuple(origins)


def _sample_id(source_hash: str, seed: int, index: int, origin: tuple[int, int]) -> str:
    payload = f"{DATASET_CONTRACT}\0{source_hash}\0{seed}\0{index}\0{origin[0]}\0{origin[1]}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:24]


def _sample_from_pair(
    pair: PairedCrop,
    source_hash: str,
    dataset_seed: int,
    pair_seed: int,
    index: int,
    origin: tuple[int, int],
    crop_size: tuple[int, int],
    source_size: tuple[int, int],
) -> PairedTrainingSample:
    source_to_crop = CoordinateTransform.translation(-origin[0], -origin[1])
    source_to_hr = source_to_crop.then(pair.source_to_hr)
    hr_to_source = source_to_hr.inverse()
    metadata = json.loads(json.dumps(pair.metadata, sort_keys=True, allow_nan=False))
    metadata["dataset"] = {
        "contract": DATASET_CONTRACT,
        "dataset_seed": dataset_seed,
        "pair_seed": pair_seed,
        "pair_index": index,
        "source_sha256": source_hash,
        "source_size": [source_size[0], source_size[1]],
        "crop": {
            "left": origin[0],
            "top": origin[1],
            "width": crop_size[0],
            "height": crop_size[1],
        },
        "source_to_hr": source_to_hr.to_json(),
        "hr_to_source": hr_to_source.to_json(),
    }
    return PairedTrainingSample(
        sample_id=_sample_id(source_hash, dataset_seed, index, origin),
        seed=pair_seed,
        hr=pair.hr,
        lr=pair.lr,
        marker_centers_hr=pair.marker_centers_hr,
        marker_centers_lr=pair.marker_centers_lr,
        metadata=metadata,
        hr_to_lr=pair.hr_to_lr,
        lr_to_hr=pair.lr_to_hr,
        source_to_hr=source_to_hr,
        hr_to_source=hr_to_source,
    )


def build_training_pairs(
    image: np.ndarray | Image.Image,
    annotation_or_centers: Mapping[str, object] | Iterable[Sequence[float]],
    *,
    seed: int,
    crop_size: int | Sequence[int] = (96, 96),
    count: int = 1,
    degradation_config: DegradationConfig | None = None,
) -> tuple[PairedTrainingSample, ...]:
    """Build deterministic, marker-aware pairs from a clean rendered graph.

    ``crop_size`` is ``(width, height)``.  Markers are selected in source-image
    coordinates and filtered into each crop without fabricating values.
    """

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if not 0 <= seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= _MAX_PAIR_COUNT:
        raise ValueError(f"count must be between 1 and {_MAX_PAIR_COUNT}")
    source = _as_rgb_uint8(image)
    image_height, image_width = source.shape[:2]
    width, height = _crop_size(crop_size, image_width, image_height)
    centers = _normalize_centers_input(annotation_or_centers, image_width, image_height)
    origins = _select_origins(image_width, image_height, width, height, centers, seed, count)
    source_hash = hashlib.sha256(source.tobytes(order="C")).hexdigest()

    samples: list[PairedTrainingSample] = []
    for index, (left, top) in enumerate(origins):
        crop = np.ascontiguousarray(source[top : top + height, left : left + width])
        crop_centers = tuple(
            (x - left, y - top)
            for x, y in centers
            if left <= x < left + width and top <= y < top + height
        )
        pair_seed = _derived_pair_seed(seed, index)
        pair = build_paired_crop(
            crop,
            crop_centers,
            pair_seed,
            degradation_config,
        )
        samples.append(
            _sample_from_pair(
                pair,
                source_hash,
                seed,
                pair_seed,
                index,
                (left, top),
                (width, height),
                (image_width, image_height),
            )
        )
    return tuple(samples)


def build_pairs_from_scene(
    scene: Mapping[str, object],
    *,
    seed: int | None = None,
    crop_size: int | Sequence[int] = (96, 96),
    count: int = 1,
    degradation_config: DegradationConfig | None = None,
) -> tuple[PairedTrainingSample, ...]:
    """Render a local ``ml.synthetic`` scene and build GraphSR pairs from it."""

    from ml.synthetic.renderer import render_scene

    rendered, annotation, _marker_mask = render_scene(dict(scene))
    effective_seed = int(scene.get("seed", 0)) if seed is None else seed
    return build_training_pairs(
        rendered,
        annotation,
        seed=effective_seed,
        crop_size=crop_size,
        count=count,
        degradation_config=degradation_config,
    )


__all__ = [
    "DATASET_CONTRACT",
    "GraphSrPairDataset",
    "PairedTrainingSample",
    "build_pairs_from_scene",
    "build_training_pairs",
    "extract_marker_centers",
]
