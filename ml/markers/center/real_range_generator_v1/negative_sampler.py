# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Deterministic, train-only stratified negative proposal sampling."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

LOW_FAINT = 0.11372548341751099
P05_FAINT = 0.1921568512916565
OCR_P95 = 0.7272727272727273
ARTIFACT_P95 = 0.12121212121212122
SAMPLER_SEED = 20260904
TOPOLOGY_RADIUS_PX = 16.0
TOPOLOGY_SAMPLER_RADIUS_PX = 12.0
TOPOLOGY_HARD_RADIUS_PX = 4.0
TOPOLOGY_KINDS = ("topology_junction", "topology_fragment")
LEGACY_HARD_KINDS = ("text", "line_intersection", "axis")
LEGACY_HARD_RADIUS_PX = 8.0
CONNECTOR_ENDPOINT_OFFSET_PX = 8.0
CONNECTOR_ANCHOR_MAX_DISTANCE_PX = 4.0
GENERIC_CONNECTOR_BAND_RADIUS_PX = 4.0
SPARSE_FRAGMENT_RADIUS_PX = 8.0
# Compatibility export for the deferred V24 trainer; endpoint selection no
# longer consumes these legacy fractional anchors.
CONNECTOR_ANCHOR_FRACTIONS = (1.0 / 3.0, 2.0 / 3.0)
QUOTAS = {
    "hard_existing": 6012,
    "faint_low": 326,
    "faint_p05": 1303,
    "ocr_heavy": 1629,
    "artifact": 1629,
    "generic_connector_band": 6720,
    "generic": 14961,
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
    connector_anchor_target_count: int = 0
    connector_anchor_capacity: int = 0
    connector_anchor_selected: int = 0
    connector_anchor_selected_index_sha256: str = ""
    connector_anchor_max_distance_px: float = CONNECTOR_ANCHOR_MAX_DISTANCE_PX
    topology_sampler_radius_px: float = TOPOLOGY_SAMPLER_RADIUS_PX
    connector_endpoint_offset_px: float = CONNECTOR_ENDPOINT_OFFSET_PX
    generic_remainder_selected: int = 0
    topology_hard_capacity: dict[str, int] | None = None
    topology_hard_selected: dict[str, int] | None = None
    hard_training_total: int = 0
    generic_connector_band_selected_index_sha256: str = ""
    sparse_fragment_capacity: int = 0
    sparse_fragment_selected: int = 0

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
    for kind, x, y in scene.hard_negatives:
        if kind not in LEGACY_HARD_KINDS:
            continue
        hard |= torch.cdist(
            coordinates, torch.tensor(((x, y),), dtype=coordinates.dtype)
        ).squeeze(1).le(LEGACY_HARD_RADIUS_PX)
    return torch.nonzero(hard & (labels <= 0.5)).flatten()


def _topology_indices(scene: Any, coordinates: torch.Tensor, labels: torch.Tensor, *, radius_px: float = TOPOLOGY_RADIUS_PX) -> dict[str, set[int]]:
    result = {kind: set() for kind in TOPOLOGY_KINDS}
    for kind, x, y in scene.hard_negatives:
        if kind not in result:
            continue
        distance = torch.linalg.vector_norm(
            coordinates - torch.tensor((x, y), dtype=coordinates.dtype), dim=1
        )
        result[kind].update(torch.nonzero((distance <= radius_px) & (labels <= 0.5)).flatten().tolist())
    return result


def _connector_anchor_indices(scene: Any, coordinates: torch.Tensor, labels: torch.Tensor) -> set[int]:
    """Select nearest negative proposals to fixed one-third/two-thirds anchors."""
    centers = tuple((float(x), float(y)) for x, y in scene.centers)
    eligible = labels <= 0.5
    selected: set[int] = set()
    for first, second in zip(centers, centers[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        length = math.hypot(dx, dy)
        if length == 0.0:
            raise ValueError("connector segment has zero length")
        unit_x, unit_y = dx / length, dy / length
        for anchor_x, anchor_y in (
            (first[0] + CONNECTOR_ENDPOINT_OFFSET_PX * unit_x, first[1] + CONNECTOR_ENDPOINT_OFFSET_PX * unit_y),
            (second[0] - CONNECTOR_ENDPOINT_OFFSET_PX * unit_x, second[1] - CONNECTOR_ENDPOINT_OFFSET_PX * unit_y),
        ):
            anchor = torch.tensor((anchor_x, anchor_y), dtype=coordinates.dtype)
            distances = torch.linalg.vector_norm(coordinates - anchor, dim=1)
            distances = torch.where(eligible, distances, torch.full_like(distances, float("inf")))
            index = int(torch.argmin(distances).item())
            if not bool(torch.isfinite(distances[index])) or float(distances[index]) > CONNECTOR_ANCHOR_MAX_DISTANCE_PX:
                raise ValueError("connector anchor lacks an eligible proposal within 4 px")
            selected.add(index)
    return selected


def _sparse_fragment_indices(scene: Any, coordinates: torch.Tensor, labels: torch.Tensor) -> set[int]:
    selected: set[int] = set()
    for kind, x, y in scene.hard_negatives:
        if kind == "sparse_fragment":
            distance = torch.linalg.vector_norm(
                coordinates - coordinates.new_tensor((x, y)), dim=1)
            selected.update(torch.nonzero(
                (distance <= SPARSE_FRAGMENT_RADIUS_PX) & (labels <= 0.5)
            ).flatten().tolist())
    return selected


def _connecting_segment_distances(scene: Any, coordinates: torch.Tensor) -> torch.Tensor:
    """Return each proposal's distance to the nearest center-to-center segment."""
    centers = torch.tensor(scene.centers, dtype=coordinates.dtype, device=coordinates.device)
    starts, vectors = centers[:-1], centers[1:] - centers[:-1]
    points = coordinates.unsqueeze(1)
    offset = points - starts.unsqueeze(0)
    lengths_squared = (vectors * vectors).sum(dim=1).clamp_min(1e-12)
    projection = (offset * vectors.unsqueeze(0)).sum(dim=2) / lengths_squared.unsqueeze(0)
    projection = projection.clamp(0.0, 1.0)
    nearest = starts.unsqueeze(0) + projection.unsqueeze(2) * vectors.unsqueeze(0)
    return torch.linalg.vector_norm(points - nearest, dim=2).amin(dim=1)


def _generic_connector_band_indices(
    scene: Any,
    coordinates: torch.Tensor,
    labels: torch.Tensor,
    features: dict[str, torch.Tensor],
    legacy_hard_indices: set[int],
    topology_by_index: dict[int, str],
    connector_indices: set[int],
) -> set[int]:
    """Select nonpositive, nonreserved proposals in the fixed connector band."""
    eligible = labels <= 0.5
    eligible &= ~torch.tensor(
        [index in legacy_hard_indices or index in topology_by_index or index in connector_indices
         for index in range(len(coordinates))],
        dtype=torch.bool,
        device=coordinates.device,
    )
    eligible &= ~(
        features["faint_low"]
        | features["faint_p05"]
        | features["ocr_heavy"]
        | features["artifact"]
    )
    eligible &= _connecting_segment_distances(scene, coordinates) <= GENERIC_CONNECTOR_BAND_RADIUS_PX
    return set(torch.nonzero(eligible, as_tuple=False).flatten().tolist())


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
    topology_hard_pools: dict[str, set[tuple[str, int, int]]] = {kind: set() for kind in TOPOLOGY_KINDS}
    hard_existing_pool: set[tuple[str, int, int]] = set()
    connector_pools: set[tuple[str, int, int]] = set()
    sparse_pools: set[tuple[str, int, int]] = set()
    connector_anchor_target_count = 0
    generic_remainder_selected = 0
    generic_connector_band_selected_index_sha256 = ""
    selections: list[list[int]] = []
    capacities = {name: 0 for name in target}
    for scene_number, (scene, proposals, labels, hard) in enumerate(records):
        if len(proposals.patches) != len(labels) or len(labels) != len(hard):
            raise ValueError("proposal, label, and hard arrays must have equal length")
        features = _features(proposals.patches)
        positive = labels > 0.5
        legacy_hard_indices = set(_hard_indices(scene, proposals.coordinates, labels).tolist())
        topology = _topology_indices(scene, proposals.coordinates, labels, radius_px=TOPOLOGY_SAMPLER_RADIUS_PX)
        topology_hard = _topology_indices(scene, proposals.coordinates, labels, radius_px=TOPOLOGY_HARD_RADIUS_PX)
        connector_indices = _connector_anchor_indices(scene, proposals.coordinates, labels)
        sparse_indices = _sparse_fragment_indices(scene, proposals.coordinates, labels)
        connector_anchor_target_count += max(0, (len(scene.centers) - 1) * 2)
        topology_by_index = {index: kind for kind in TOPOLOGY_KINDS for index in topology[kind]}
        generic_connector_band_indices = _generic_connector_band_indices(
            scene, proposals.coordinates, labels, features, legacy_hard_indices,
            topology_by_index, connector_indices)
        for index in torch.nonzero(~positive, as_tuple=False).flatten().tolist():
            if index in legacy_hard_indices:
                hard_existing_pool.add((_stable_key(seed, split, int(scene.seed), index, "hard_existing"), scene_number, index))
            if index in legacy_hard_indices:
                name = "hard_existing"
            elif index in topology_by_index or index in connector_indices:
                name = "generic"
            elif bool(features["faint_low"][index]):
                name = "faint_low"
            elif bool(features["faint_p05"][index]):
                name = "faint_p05"
            elif bool(features["ocr_heavy"][index]):
                name = "ocr_heavy"
            elif bool(features["artifact"][index]):
                name = "artifact"
            elif index in generic_connector_band_indices:
                name = "generic_connector_band"
            else:
                name = "generic"
            pools[name].append((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            if index in sparse_indices and name == "generic":
                sparse_pools.add((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            for kind in TOPOLOGY_KINDS:
                if index in topology[kind]:
                    topology_pools[kind].add((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
                if index in topology_hard[kind]:
                    topology_hard_pools[kind].add((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            if index in connector_indices:
                connector_pools.add((_stable_key(seed, split, int(scene.seed), index, name), scene_number, index))
            if name != "hard_existing":
                capacities[name] += 1
            selections.append([])
    capacities["hard_existing"] = len(hard_existing_pool)
    for name, quota in target.items():
        if capacities[name] < quota:
            raise ValueError(f"negative stratum under-capacity: {name} {capacities[name]} < {quota}")
    if capacities["hard_existing"] > target["hard_existing"]:
        raise ValueError("hard-negative capacity exceeds fixed quota")
    if capacities["hard_existing"] != target["hard_existing"]:
        raise ValueError("all existing hard negatives must be retained")
    selected_topology: dict[str, int] = {kind: 0 for kind in TOPOLOGY_KINDS}
    selected_hard = sorted(hard_existing_pool)[: target["hard_existing"]]
    for _, scene_number, index in selected_hard:
        selections[scene_number].append(index)
    selected_hard_pairs = {(scene_number, index) for _, scene_number, index in selected_hard}
    generic_entries = set(pools["generic"])
    topology_entries = {
        entry for entry in (set().union(*topology_pools.values()) if topology_pools else set()) & generic_entries
        if (entry[1], entry[2]) not in selected_hard_pairs
    }
    connector_entries = {
        entry for entry in connector_pools & generic_entries
        if (entry[1], entry[2]) not in selected_hard_pairs
    }
    retained_generic_entries = topology_entries | connector_entries | sparse_pools
    if len(retained_generic_entries) > target["generic"]:
        raise ValueError("topology plus connector and sparse coverage exceeds generic quota")
    for name, quota in target.items():
        if name == "hard_existing":
            continue
        if name == "generic":
            ranked_topology = sorted(topology_entries)
            ranked_connectors = sorted(connector_entries - topology_entries)
            ranked_sparse = sorted(sparse_pools - topology_entries - connector_entries)
            generic_remaining = [item for item in sorted(pools[name]) if item not in retained_generic_entries and (item[1], item[2]) not in selected_hard_pairs]
            generic_remainder = generic_remaining[: quota - len(ranked_topology) - len(ranked_connectors) - len(ranked_sparse)]
            generic_remainder_selected = len(generic_remainder)
            ranked = ranked_topology + ranked_connectors + ranked_sparse + generic_remainder
        elif name == "generic_connector_band":
            ranked = sorted(pools[name])[:quota]
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
    selected_topology_hard_pairs = {
        kind: {
            (scene_number, index)
            for _, scene_number, index in topology_hard_pools[kind]
            if index in selected_sets[scene_number]
        }
        for kind in TOPOLOGY_KINDS
    }
    selected_topology_hard = {kind: len(pairs) for kind, pairs in selected_topology_hard_pairs.items()}
    topology_hard_capacity = {kind: len(topology_hard_pools[kind]) for kind in TOPOLOGY_KINDS}
    if selected_topology_hard != topology_hard_capacity:
        raise ValueError("hard topology proposal was not selected")
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
        for _, scene_number, index in sorted(topology_pools[kind], key=lambda item: (item[1], item[2])):
            if index in selected[scene_number]:
                topology_digest.update(f"{kind}:{scene_number}:{index}\n".encode("ascii"))
    connector_digest = hashlib.sha256()
    connector_selected = 0
    for _, scene_number, index in sorted(connector_pools):
        if index in selected_sets[scene_number]:
            connector_selected += 1
            connector_digest.update(f"{scene_number}:{index}\n".encode("ascii"))
    generic_connector_band_digest = hashlib.sha256()
    for _, scene_number, index in sorted(
        pools["generic_connector_band"], key=lambda item: (item[1], item[2])
    ):
        if index in selected_sets[scene_number]:
            generic_connector_band_digest.update(f"{scene_number}:{index}\n".encode("ascii"))
    generic_connector_band_selected_index_sha256 = generic_connector_band_digest.hexdigest()
    return SampledNegatives(
        selected, capacities, counts, digest.hexdigest(),
        {kind: len(topology_pools[kind]) for kind in TOPOLOGY_KINDS},
        selected_topology,
        topology_digest.hexdigest(),
        connector_anchor_target_count,
        len(connector_pools),
        connector_selected,
        connector_digest.hexdigest(),
        CONNECTOR_ANCHOR_MAX_DISTANCE_PX,
        TOPOLOGY_SAMPLER_RADIUS_PX,
        CONNECTOR_ENDPOINT_OFFSET_PX,
        generic_remainder_selected,
        topology_hard_capacity,
        selected_topology_hard,
        len(selected_hard_pairs | set().union(*selected_topology_hard_pairs.values())),
        generic_connector_band_selected_index_sha256,
        len(sparse_pools),
        sum(index in selected_sets[scene_number] for _, scene_number, index in sparse_pools),
    )
