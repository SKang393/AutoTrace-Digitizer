# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic, train-only stratified negative proposal sampling."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import torch

LOW_FAINT = 0.11372548341751099
P05_FAINT = 0.1921568512916565
OCR_P95 = 0.7272727272727273
ARTIFACT_P95 = 0.12121212121212122
SAMPLER_SEED = 20260904
TOPOLOGY_RADIUS_PX = 16.0
TOPOLOGY_KINDS = ("topology_junction", "topology_fragment")
QUOTAS = {
    "hard_existing": 6012,
    "faint_low": 326,
    "faint_p05": 1303,
    "ocr_heavy": 1629,
    "artifact": 1629,
    "generic": 21681,
}


@dataclass(frozen=True)
class SampledNegatives:
    selections: tuple[tuple[int, ...], ...]
    capacities: dict[str, int]
    counts: dict[str, int]
    selected_index_sha256: str
    topology_capacity: dict[str, int] | None = None
    topology_selected: dict[str, int] | None = None
    topology_selected_index_sha256: str = ""

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _stable_key(seed: int, split: str, scene_seed: int, index: int, stratum: str) -> str:
    payload = f"{seed}|{split}|{scene_seed}|{index}|{stratum}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _features(patches: torch.Tensor) -> dict[str, torch.Tensor]:
    middle = patches.shape[-1] // 2
    return {
        "ink_max": patches[:, 0].amax(dim=(1, 2)),
        "ocr_mean": patches[:, 1].mean(dim=(1, 2)),
        "artifact_mean": patches[:, 2].mean(dim=(1, 2)),
        "faint_low": patches[:, 0].amax(dim=(1, 2)).le(LOW_FAINT),
        "faint_p05": patches[:, 0].amax(dim=(1, 2)).gt(LOW_FAINT)
        & patches[:, 0].amax(dim=(1, 2)).le(P05_FAINT),
        "ocr_heavy": patches[:, 1].mean(dim=(1, 2)).ge(OCR_P95),
        "artifact": patches[:, 2].mean(dim=(1, 2)).ge(ARTIFACT_P95),
    }


def _hard_indices(scene: Any, coordinates: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    hard = torch.zeros(len(coordinates), dtype=torch.bool)
    for _, x, y in scene.hard_negatives:
        hard |= torch.cdist(
            coordinates, torch.tensor(((x, y),), dtype=coordinates.dtype)
        ).squeeze(1).le(8.0)
    return torch.nonzero(hard & (labels <= 0.5)).flatten()


def _topology_indices(scene: Any, coordinates: torch.Tensor, labels: torch.Tensor) -> dict[str, set[int]]:
    result = {kind: set() for kind in TOPOLOGY_KINDS}
    for kind, x, y in scene.hard_negatives:
        if kind not in result:
            continue
        distance = torch.linalg.vector_norm(
            coordinates - torch.tensor((x, y), dtype=coordinates.dtype), dim=1
        )
        result[kind].update(torch.nonzero((distance <= TOPOLOGY_RADIUS_PX) & (labels <= 0.5)).flatten().tolist())
    return result


def sample_negatives(
    records: Iterable[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    split: str,
    seed: int = SAMPLER_SEED,
    quotas: dict[str, int] | None = None,
) -> SampledNegatives:
    """Select exact global quotas from ``(scene, proposals, labels, hard)`` records.

    The caller supplies proposal batches and labels, keeping this utility independent
    of model code. It is intended for ``train`` only and fails closed on every
    contract violation.
    """
    if split != "train":
        raise ValueError("negative sampling is train-only")
    target = dict(QUOTAS if quotas is None else quotas)
    if sum(target.values()) != 32580:
        raise ValueError("negative quotas must total 32580")
    pools: dict[str, list[tuple[str, int, int]]] = {name: [] for name in target}
    topology_pools: dict[str, set[tuple[str, int, int]]] = {kind: set() for kind in TOPOLOGY_KINDS}
    selections: list[list[int]] = []
    capacities = {name: 0 for name in target}
    for scene_number, (scene, proposals, labels, hard) in enumerate(records):
        if len(proposals.patches) != len(labels) or len(labels) != len(hard):
            raise ValueError("proposal, label, and hard arrays must have equal length")
        features = _features(proposals.patches)
        positive = labels > 0.5
        hard_mask = hard & ~positive
        topology = _topology_indices(scene, proposals.coordinates, labels)
        topology_by_index = {index: kind for kind in TOPOLOGY_KINDS for index in topology[kind]}
        for index in torch.nonzero(~positive, as_tuple=False).flatten().tolist():
            if bool(hard_mask[index]):
                name = "hard_existing"
            elif index in topology_by_index:
                name = "generic"
            elif bool(features["faint_low"][index]):
                name = "faint_low"
            elif bool(features["faint_p05"][index]):
                name = "faint_p05"
            elif bool(features["ocr_heavy"][index]):
                name = "ocr_heavy"
            elif bool(features["artifact"][index]):
                name = "artifact"
            else:
                name = "generic"
            pools[name].append((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            for kind in TOPOLOGY_KINDS:
                if index in topology[kind]:
                    topology_pools[kind].add((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            capacities[name] += 1
        selections.append([])
    for name, quota in target.items():
        if capacities[name] < quota:
            raise ValueError(f"negative stratum under-capacity: {name} {capacities[name]} < {quota}")
    if capacities["hard_existing"] > target["hard_existing"]:
        raise ValueError("hard-negative capacity exceeds fixed quota")
    if capacities["hard_existing"] != target["hard_existing"]:
        raise ValueError("all existing hard negatives must be retained")
    selected_topology: dict[str, int] = {kind: 0 for kind in TOPOLOGY_KINDS}
    generic_entries = set(pools["generic"])
    topology_entries = (set().union(*topology_pools.values()) if topology_pools else set()) & generic_entries
    if len(topology_entries) > target["generic"]:
        raise ValueError("topology coverage exceeds generic quota")
    for name, quota in target.items():
        if name == "generic":
            ranked_topology = sorted(topology_entries)
            generic_remaining = [item for item in sorted(pools[name]) if item not in topology_entries]
            ranked = ranked_topology + generic_remaining[: quota - len(ranked_topology)]
        else:
            ranked = sorted(pools[name])[:quota]
        for _, scene_number, index in ranked:
            selections[scene_number].append(index)
    selected = tuple(tuple(sorted(indices)) for indices in selections)
    selected_sets = [set(indices) for indices in selected]
    for kind in TOPOLOGY_KINDS:
        selected_topology[kind] = sum(
            int(index in selected_sets[scene_number])
            for _, scene_number, index in topology_pools[kind]
        )
    counts = {name: target[name] for name in target}
    if counts["hard_existing"] != sum(
        len(indices) for indices in selected
    ) - sum(counts[name] for name in target if name != "hard_existing"):
        raise ValueError("hard-negative quota mismatch")
    digest = hashlib.sha256()
    for scene_number, indices in enumerate(selected):
        for index in indices:
            digest.update(f"{scene_number}:{index}\n".encode("ascii"))
    if sum(counts.values()) != 32580:
        raise ValueError("selected negative total mismatch")
    topology_digest = hashlib.sha256()
    for kind in TOPOLOGY_KINDS:
        for _, scene_number, index in sorted(topology_pools[kind]):
            if index in selected[scene_number]:
                topology_digest.update(f"{kind}:{scene_number}:{index}\n".encode("ascii"))
    return SampledNegatives(
        selected, capacities, counts, digest.hexdigest(),
        {kind: len(topology_pools[kind]) for kind in TOPOLOGY_KINDS},
        selected_topology,
        topology_digest.hexdigest(),
    )
