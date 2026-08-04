# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Fixed P2 selection split with procedural short tick and joint hard negatives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from .dataset import Scene, build_fixed_dataset


DATASET_REVISION = "marker-center-procedural-v2-p2-tick-joint"
P2_SPLIT_FAMILIES = {
    "train": ("vector_clean", "print_speckle", "short_tick_joint_train"),
    "validation": ("scan_gaussian", "short_tick_joint_validation"),
}
P2_VARIANTS = {"train": 4, "validation": 3}
P2_SEED_BASE = {"train": 4100, "validation": 4200}


def _draw_short_structure(tensor: torch.Tensor, x: int, y: int, *, joint: bool) -> None:
    ink = tensor[0]
    artifact = tensor[2]
    if joint:
        ink[y - 1 : y + 2, x - 5 : x + 6] = torch.maximum(
            ink[y - 1 : y + 2, x - 5 : x + 6],
            torch.tensor(0.92, dtype=ink.dtype),
        )
        ink[y - 5 : y + 6, x - 1 : x + 2] = torch.maximum(
            ink[y - 5 : y + 6, x - 1 : x + 2],
            torch.tensor(0.92, dtype=ink.dtype),
        )
    else:
        ink[y - 5 : y + 6, x - 1 : x + 2] = torch.maximum(
            ink[y - 5 : y + 6, x - 1 : x + 2],
            torch.tensor(0.92, dtype=ink.dtype),
        )
        ink[y - 5 : y - 2, x - 5 : x + 6] = torch.maximum(
            ink[y - 5 : y - 2, x - 5 : x + 6],
            torch.tensor(0.92, dtype=ink.dtype),
        )
    artifact[y - 7 : y + 8, x - 7 : x + 8] = 1.0


def _tick_joint_scene(split: str, variant: int) -> Scene:
    base = build_fixed_dataset(split)[variant]
    tensor = base.tensor.clone()
    additions: list[tuple[str, float, float]] = []
    for index, x in enumerate((20, 38, 56, 74, 92, 110)):
        y = 84 + 3 * ((index + variant) % 3)
        joint = (index + variant) % 2 == 0
        _draw_short_structure(tensor, x, y, joint=joint)
        additions.append(("line_intersection" if joint else "tick", float(x), float(y)))
    artifact_target = torch.maximum(base.artifact_target, tensor[2:3])
    family = f"short_tick_joint_{split}"
    return Scene(
        scene_id=f"{split}-{family}-{variant}",
        split=split,
        family=family,
        degradation="procedural_short_perpendicular_tick_joint",
        seed=P2_SEED_BASE[split] + variant,
        tensor=tensor,
        center_target=base.center_target.clone(),
        radius_target=base.radius_target.clone(),
        artifact_target=artifact_target,
        centers=base.centers,
        radii=base.radii,
        hard_negatives=base.hard_negatives + tuple(additions),
    )


def build_p2_selection_dataset(split: str) -> tuple[Scene, ...]:
    if split not in P2_SPLIT_FAMILIES:
        raise ValueError(f"Unknown P2 selection split {split!r}")
    base = build_fixed_dataset(split)
    added = tuple(_tick_joint_scene(split, variant) for variant in range(P2_VARIANTS[split]))
    return base + added


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def p2_dataset_manifest() -> dict[str, object]:
    cases = []
    for split in ("train", "validation"):
        for scene in build_p2_selection_dataset(split):
            cases.append(
                {
                    "scene_id": scene.scene_id,
                    "split": split,
                    "family": scene.family,
                    "degradation": scene.degradation,
                    "seed": scene.seed,
                    "tensor_sha256": _tensor_sha256(scene.tensor),
                    "center_target_sha256": _tensor_sha256(scene.center_target),
                    "radius_target_sha256": _tensor_sha256(scene.radius_target),
                    "artifact_target_sha256": _tensor_sha256(scene.artifact_target),
                    "center_count": len(scene.centers),
                    "hard_negative_kinds": [item[0] for item in scene.hard_negatives],
                }
            )
    return {
        "manifest_version": 1,
        "dataset_revision": DATASET_REVISION,
        "selection_only": True,
        "public_split_included": False,
        "private_data": False,
        "split_families": P2_SPLIT_FAMILIES,
        "variants": P2_VARIANTS,
        "seed_base": P2_SEED_BASE,
        "isolated_change": "add short perpendicular tick and joint hard-negative families",
        "cases": cases,
    }


def seal_p2_dataset_manifest(output_dir: Path) -> tuple[Path, str]:
    payload = json.dumps(p2_dataset_manifest(), indent=2, sort_keys=True) + "\n"
    encoded = payload.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "dataset-manifest.json"
    path.write_bytes(encoded)
    (output_dir / "dataset-manifest.sha256").write_text(f"{digest}  dataset-manifest.json\n", encoding="ascii")
    return path, digest


__all__ = [
    "DATASET_REVISION",
    "P2_SEED_BASE",
    "P2_SPLIT_FAMILIES",
    "P2_VARIANTS",
    "build_p2_selection_dataset",
    "p2_dataset_manifest",
    "seal_p2_dataset_manifest",
]
