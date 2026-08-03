# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Bounded deterministic GraphSR-x2 training for local synthetic pairs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional

from .losses import GraphSRLoss, LossWeights, MAX_ADVERSARIAL_WEIGHT
from .model import GraphSRConfig, GraphSRx2, save_checkpoint


TRAINING_REVISION = "graphsr-x2-pytorch-v1"
DEFAULT_SEED = 20260803
MAX_EPOCHS = 10_000
MAX_STEPS = 1_000_000


class TrainingCancelledError(RuntimeError):
    """Raised at a safe batch boundary when local training is cancelled."""


@dataclass(frozen=True)
class TrainingBatch:
    sample_id: str
    lr: Tensor
    hr: Tensor
    marker_center_map: Tensor
    ocr_mask: Tensor


def configure_determinism(seed: int = DEFAULT_SEED) -> None:
    """Configure deterministic CPU behavior without downloading any resources."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _to_chw_float(value: Any, name: str) -> Tensor:
    tensor = (
        value.detach().cpu()
        if isinstance(value, Tensor)
        else torch.as_tensor(np.array(value, copy=True))
    )
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be an HWC or CHW image")
    if tensor.shape[0] in (1, 3):
        chw = tensor
    elif tensor.shape[-1] in (1, 3):
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"{name} must contain one or three channels")
    if chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1)
    if chw.dtype == torch.uint8:
        chw = chw.float().div(255.0)
    else:
        chw = chw.float()
    if not bool(torch.isfinite(chw).all()):
        raise ValueError(f"{name} contains a non-finite value")
    minimum = float(chw.min())
    maximum = float(chw.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"{name} values must be in [0, 1]")
    return chw.contiguous()


def _field(sample: Any, name: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(name, default)
    return getattr(sample, name, default)


def _centers_to_map(centers: Any, height: int, width: int) -> Tensor:
    heatmap = torch.zeros((1, height, width), dtype=torch.float32)
    if centers is None:
        return heatmap
    yy = torch.arange(height, dtype=torch.float32).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32).view(1, width)
    for center in centers:
        if isinstance(center, Mapping):
            x, y = float(center["x"]), float(center["y"])
        elif hasattr(center, "x") and hasattr(center, "y"):
            x, y = float(center.x), float(center.y)
        else:
            x, y = float(center[0]), float(center[1])
        heatmap[0] = torch.maximum(heatmap[0], torch.exp(-((xx - x).square() + (yy - y).square()) / 32.0))
    return heatmap


def _mask_from_sample(sample: Any, name: str, height: int, width: int) -> Tensor:
    value = _field(sample, name)
    if value is None:
        metadata = _field(sample, "metadata", {})
        if isinstance(metadata, Mapping):
            value = metadata.get(name)
    if value is None:
        return torch.zeros((1, height, width), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(value)).float()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.permute(2, 0, 1)
    if mask.shape != (1, height, width):
        raise ValueError(f"{name} must match the HR image dimensions")
    if float(mask.max()) > 1.0:
        mask = mask.div(255.0)
    return mask.clamp(0.0, 1.0).contiguous()


def _prepare_sample(sample: Any, index: int) -> TrainingBatch:
    lr = _to_chw_float(_field(sample, "lr"), "lr")
    hr = _to_chw_float(_field(sample, "hr"), "hr")
    if hr.shape[-2] != lr.shape[-2] * 2 or hr.shape[-1] != lr.shape[-1] * 2:
        raise ValueError("Every training pair must have an exact x2 HR/LR spatial relationship")
    height, width = hr.shape[-2:]
    marker_center_map = _mask_from_sample(sample, "marker_center_map", height, width)
    if not bool(torch.any(marker_center_map)):
        marker_center_map = _centers_to_map(_field(sample, "marker_centers_hr"), height, width)
    ocr_mask = _mask_from_sample(sample, "ocr_mask", height, width)
    sample_id = str(_field(sample, "sample_id", f"sample-{index:04d}"))
    return TrainingBatch(sample_id, lr, hr, marker_center_map, ocr_mask)


def _draw_smoke_target(index: int, size: int = 32) -> TrainingBatch:
    """Create a tiny deterministic chart crop for import and NaN smoke probes."""

    hr = torch.ones((3, size, size), dtype=torch.float32)
    baseline_y = 23 - index
    hr[:, baseline_y : baseline_y + 1, 3:29] = 0.08
    hr[:, 4:25, 4:5] = 0.08
    center_x, center_y = 15 + index, 13 + (index % 2)
    yy = torch.arange(size).view(size, 1)
    xx = torch.arange(size).view(1, size)
    radius = torch.sqrt((xx - center_x).float().square() + (yy - center_y).float().square())
    ring = (radius >= 3.0) & (radius <= 4.25)
    hr[:, ring] = 0.04
    # Text-like strokes provide a deterministic OCR-consistency region.
    hr[:, 26:27, 8:12] = 0.12
    hr[:, 27:30, 9:10] = 0.12
    ocr_mask = torch.zeros((1, size, size), dtype=torch.float32)
    ocr_mask[:, 24:31, 6:14] = 1.0
    marker_center_map = _centers_to_map(((center_x, center_y),), size, size)
    lr = functional.interpolate(hr.unsqueeze(0), scale_factor=0.5, mode="area")[0]
    return TrainingBatch(f"built-in-smoke-{index}", lr, hr, marker_center_map, ocr_mask)


def _clean_session07_chart() -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    """Build a deterministic clean chart used only as input to the real degrader."""

    size = 96
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[76:78, 10:88] = 20
    image[8:78, 10:12] = 20
    centers = ((25.0, 59.0), (45.0, 43.0), (67.0, 52.0))
    for center_x, center_y in centers:
        yy, xx = np.ogrid[:size, :size]
        radius = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        ring = (radius >= 3.0) & (radius <= 4.2)
        image[ring] = 12
    image[20:22, 18:38] = 30
    image[20:34, 18:20] = 30
    image[32:34, 18:38] = 30
    return image, centers


def _build_degraded_synthetic_pairs(seed: int, count: int) -> tuple[Any, ...]:
    if count < 1 or count > 10_000:
        raise ValueError("synthetic_samples must be in the range 1 through 10000")
    try:
        from .dataset import build_training_pairs
    except ImportError as error:
        raise RuntimeError("The Session 07 two-stage degradation dataset is unavailable") from error
    image, centers = _clean_session07_chart()
    pairs = build_training_pairs(
        image,
        centers,
        seed=seed,
        crop_size=(96, 96),
        count=count,
    )
    annotated: list[dict[str, Any]] = []
    for pair in pairs:
        height, width = pair.hr.shape[:2]
        ocr_mask = np.zeros((height, width), dtype=np.float32)
        # The clean chart above deliberately renders its text-like glyph here.
        ocr_mask[18:36, 16:40] = 1.0
        annotated.append(
            {
                "sample_id": pair.sample_id,
                "lr": pair.lr,
                "hr": pair.hr,
                "marker_centers_hr": pair.marker_centers_hr,
                "ocr_mask": ocr_mask,
                "metadata": pair.metadata,
            }
        )
    return tuple(annotated)


def _dataset_identity(samples: tuple[TrainingBatch, ...]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        for tensor in (sample.lr, sample.hr, sample.marker_center_map, sample.ocr_mask):
            digest.update(tensor.contiguous().numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _validate_limits(epochs: int, max_steps: int, batch_size: int) -> None:
    if epochs < 1 or epochs > MAX_EPOCHS:
        raise ValueError(f"epochs must be in the range 1 through {MAX_EPOCHS}")
    if max_steps < 1 or max_steps > MAX_STEPS:
        raise ValueError(f"max_steps must be in the range 1 through {MAX_STEPS}")
    if batch_size < 1 or batch_size > 1024:
        raise ValueError("batch_size must be in the range 1 through 1024")


def train(
    output_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    epochs: int = 20,
    max_steps: int = 200,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    samples: Iterable[Any] | None = None,
    config: GraphSRConfig | None = None,
    loss_weights: LossWeights | None = None,
    stop_requested: Callable[[], bool] | None = None,
    synthetic_samples: int = 0,
) -> tuple[Path, dict[str, object]]:
    """Train on public/synthetic pairs and keep weights in an ignored project root."""

    started = time.perf_counter()
    _validate_limits(epochs, max_steps, batch_size)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    configure_determinism(seed)
    if samples is not None and synthetic_samples:
        raise ValueError("Pass either samples or synthetic_samples, not both")
    if synthetic_samples:
        generated = _build_degraded_synthetic_pairs(seed, synthetic_samples)
        prepared = tuple(_prepare_sample(sample, index) for index, sample in enumerate(generated))
        dataset_scope = "Session 07 deterministic two-stage synthetic degradation pairs"
    elif samples is None:
        prepared = tuple(_draw_smoke_target(index) for index in range(4))
        dataset_scope = "deterministic built-in synthetic smoke crops"
    else:
        prepared = tuple(_prepare_sample(sample, index) for index, sample in enumerate(samples))
        dataset_scope = "caller-supplied local synthetic pairs"
    if not prepared:
        raise ValueError("At least one training pair is required")
    shapes = {(tuple(item.lr.shape), tuple(item.hr.shape)) for item in prepared}
    if len(shapes) != 1:
        raise ValueError("All pairs in one training run must have equal crop dimensions")

    weights = loss_weights or LossWeights()
    if weights.adversarial > MAX_ADVERSARIAL_WEIGHT:
        raise ValueError("Adversarial objective exceeds the project safety limit")
    model = GraphSRx2(config or GraphSRConfig(seed=seed))
    criterion = GraphSRLoss(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history: list[dict[str, float | int]] = []
    step = 0
    model.train()
    for epoch in range(epochs):
        if stop_requested is not None and stop_requested():
            raise TrainingCancelledError("GraphSR training cancelled before the next epoch")
        order = torch.randperm(len(prepared), generator=generator).tolist()
        for start in range(0, len(order), batch_size):
            if step >= max_steps:
                break
            if stop_requested is not None and stop_requested():
                raise TrainingCancelledError("GraphSR training cancelled at a batch boundary")
            selected = [prepared[index] for index in order[start : start + batch_size]]
            lr_batch = torch.stack([item.lr for item in selected])
            hr_batch = torch.stack([item.hr for item in selected])
            marker_batch = torch.stack([item.marker_center_map for item in selected])
            ocr_batch = torch.stack([item.ocr_mask for item in selected])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lr_batch)
            components = criterion(prediction, hr_batch, marker_batch, ocr_batch)
            if not all(bool(torch.isfinite(value).all()) for value in components.values()):
                raise FloatingPointError("GraphSR loss became non-finite")
            components["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("GraphSR gradient norm became non-finite")
            optimizer.step()
            step += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "step": step,
                    **{name: float(value.detach()) for name, value in components.items()},
                    "gradient_norm": float(gradient_norm),
                }
            )
        if step >= max_steps:
            break

    model.eval()
    with torch.inference_mode():
        validation_input = prepared[0].lr.unsqueeze(0)
        validation_output = model(validation_input)
    if not bool(torch.isfinite(validation_output).all()):
        raise FloatingPointError("GraphSR validation output became non-finite")

    identity = _dataset_identity(prepared)
    output_dir = output_dir.expanduser().resolve()
    checkpoint = save_checkpoint(
        output_dir / "graphsr-x2.pt",
        model,
        dataset_identity=identity,
        training_revision=TRAINING_REVISION,
        loss_weights=asdict(weights),
        seed=seed,
    )
    report: dict[str, object] = {
        "status": "trained",
        "training_revision": TRAINING_REVISION,
        "architecture": model.config.architecture,
        "config": asdict(model.config),
        "tensor_contract": asdict(model.contract),
        "seed": seed,
        "epochs_requested": epochs,
        "steps_completed": step,
        "max_steps": max_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "dataset_scope": dataset_scope,
        "dataset_identity": identity,
        "sample_count": len(prepared),
        "loss_weights": asdict(weights),
        "adversarial_objective_implemented": False,
        "adversarial_weight_limit": MAX_ADVERSARIAL_WEIGHT,
        "finite_loss_history": all(
            np.isfinite(value)
            for item in history
            for key, value in item.items()
            if key not in ("epoch", "step")
        ),
        "loss_history": history,
        "input_bounds": [float(validation_input.min()), float(validation_input.max())],
        "output_bounds": [float(validation_output.min()), float(validation_output.max())],
        "heldout_test_evaluations": 0,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=8,
        help="Number of deterministic two-stage pairs from ml.graphsr.dataset",
    )
    args = parser.parse_args()
    _, report = train(
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        synthetic_samples=args.synthetic_samples,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_SEED",
    "TRAINING_REVISION",
    "TrainingBatch",
    "TrainingCancelledError",
    "configure_determinism",
    "train",
]
