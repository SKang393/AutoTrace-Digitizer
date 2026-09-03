# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Aggregate-only fixed-threshold diagnosis for the V24 retry model."""

from __future__ import annotations

import argparse, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ml.markers.center.real_range_generator_v1.generator import build_split
from ml.markers.center.real_range_generator_v1.negative_sampler import _features
from ..mask_preserving import extract_proposals, postprocess, prohibited_hits

ROOT = Path(__file__).resolve().parents[5]
THRESHOLD = 0.25
TOLERANCE = 5.0
STRATA = ("hard_existing", "faint_low", "faint_p05", "ocr_heavy", "artifact", "generic")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .05, .50, .90, .95, 1])
    return {"count": len(values), "minimum": float(q[0]), "p05": float(q[1]),
            "median": float(q[2]), "p90": float(q[3]), "p95": float(q[4]),
            "maximum": float(q[5])}


def _labels(scene, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.tensor(scene.centers, dtype=torch.float32)
    distance, _ = torch.cdist(coordinates, centers).min(dim=1)
    labels = distance.le(3.0)
    hard = torch.zeros(len(coordinates), dtype=torch.bool)
    for kind, x, y in scene.hard_negatives:
        if kind in {"text", "line_intersection", "axis"}:
            hard |= torch.cdist(coordinates, torch.tensor(((x, y),), dtype=torch.float32)).squeeze(1).le(8.0)
    return labels, hard


def _strata(patches: torch.Tensor, hard: torch.Tensor, labels: torch.Tensor) -> list[str | None]:
    features = _features(patches)
    result: list[str | None] = []
    for i in range(len(patches)):
        if bool(labels[i]):
            result.append(None)
        elif bool(hard[i]):
            result.append("hard_existing")
        elif bool(features["faint_low"][i]):
            result.append("faint_low")
        elif bool(features["faint_p05"][i]):
            result.append("faint_p05")
        elif bool(features["ocr_heavy"][i]):
            result.append("ocr_heavy")
        elif bool(features["artifact"][i]):
            result.append("artifact")
        else:
            result.append("generic")
    return result


def _matched_truths(predictions, centers):
    edges = sorted((math.hypot(p.x - x, p.y - y), i, j)
                   for i, p in enumerate(predictions)
                   for j, (x, y) in enumerate(centers)
                   if math.hypot(p.x - x, p.y - y) <= TOLERANCE)
    used_p, used_t = set(), set()
    for _, i, j in edges:
        if i not in used_p and j not in used_t:
            used_p.add(i); used_t.add(j)
    return used_p, used_t


def summarize(model_path: Path) -> dict:
    started = time.perf_counter()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_hash = _sha(model_path)
    expected_model = "aee3b4ba47197d84eeecacc631730e67fd99c261b913643f95c25a8ea2436c11"
    if model_hash != expected_model:
        raise ValueError(f"retry ONNX hash mismatch: {model_hash}")
    sampler_path = ROOT / "ml/markers/center/real_range_generator_v1/negative_sampler.py"
    audit_path = ROOT / "ml/markers/center/real_range_generator_v1/AUDIT.json"
    config_path = ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json"
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider":
        raise RuntimeError("CPUExecutionProvider was not selected")
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    scenes = build_split("dev")
    strata_values: dict[str, list[float]] = {name: [] for name in STRATA}
    strata_above = defaultdict(int); strata_capacity = defaultdict(int)
    positives: list[float] = []
    scenario = {name: {"truth": 0, "tp": 0, "fn": 0} for name in ("unmasked", "ocr_mask", "artifact_mask", "both_masks")}
    accepted_fp = defaultdict(int); totals = defaultdict(int)
    for scene in scenes:
        proposals = extract_proposals(scene.tensor)
        labels, hard = _labels(scene, proposals.coordinates)
        names = _strata(proposals.patches, hard, labels)
        output = session.run([output_name], {input_name: proposals.patches.numpy().astype(np.float32, copy=False)})[0]
        probabilities = output[:, 0].astype(np.float64)
        for i, name in enumerate(names):
            if name is None:
                positives.append(float(probabilities[i]))
            else:
                strata_capacity[name] += 1; strata_values[name].append(float(probabilities[i]))
                strata_above[name] += int(probabilities[i] >= THRESHOLD)
        predictions = postprocess(scene, proposals, output)
        used_p, used_t = _matched_truths(predictions, scene.centers)
        totals["truth"] += len(scene.centers); totals["accepted"] += len(predictions)
        totals["tp"] += len(used_t); totals["fp"] += len(predictions) - len(used_t); totals["fn"] += len(scene.centers) - len(used_t)
        totals["prohibited"] += sum(prohibited_hits(predictions, scene).values())
        # Attribute each accepted false positive to its closest above-threshold proposal.
        decoded = [(float(x + output[i, 1] * 4.0), float(y + output[i, 2] * 4.0), i)
                   for i, (x, y) in enumerate(proposals.coordinates.tolist()) if probabilities[i] >= THRESHOLD]
        for pi, prediction in enumerate(predictions):
            if pi in used_p:
                continue
            if decoded:
                _, _, source = min(decoded, key=lambda item: math.hypot(prediction.x-item[0], prediction.y-item[1]))
                name = names[source]
                accepted_fp[name or "unattributed"] += 1
            else:
                accepted_fp["unattributed"] += 1
        for local, center in enumerate(scene.centers):
            x, y = (int(round(value)) for value in center)
            ocr = float(scene.tensor[1, max(0, y-2):y+3, max(0, x-2):x+3].max()) >= .35
            artifact = float(scene.tensor[2, max(0, y-2):y+3, max(0, x-2):x+3].max()) >= .35
            key = "both_masks" if ocr and artifact else "ocr_mask" if ocr else "artifact_mask" if artifact else "unmasked"
            scenario[key]["truth"] += 1; scenario[key]["tp"] += int(local in used_t); scenario[key]["fn"] += int(local not in used_t)
    for value in scenario.values():
        value["recall"] = value["tp"] / max(1, value["truth"])
    total_accepted = totals["tp"] + totals["fp"]
    report = {
        "schema": "graphreader.marker-center-mask-preserving-v24-retry-diagnosis.v1",
        "revision": "marker-center-mask-preserving-v24",
        "scope": {"synthetic_only": True, "split": "real-range-generator-v1-dev", "scene_count": len(scenes),
                  "truth_count": totals["truth"], "label_positive_distance_px": 3.0, "threshold": THRESHOLD,
                  "private_data": False, "real_dev_reads": 0, "real_sealed_reads": 0, "optimizer_steps": 0,
                  "case_ids_or_pixels_emitted": False},
        "binding": {"model_path": model_path.name, "model_sha256": model_hash, "provider": session.get_providers()[0],
                    "input_shape": ["candidate_count", 3, 33, 33], "output_shape": ["candidate_count", 4],
                    "generator_audit_sha256": _sha(audit_path), "generator_dev_split_sha256": json.loads(config_path.read_text())["dev_split_sha256"],
                    "negative_sampler_sha256": _sha(sampler_path), "negative_sampler_seed": 20260904,
                    "negative_sampler_priority": list(STRATA)},
        "proposals": {"positive_raw_probability": _quantiles(positives),
                      "negative_total": sum(strata_capacity.values()),
                      "negative_strata": {name: {"capacity": strata_capacity[name], "raw_probability": _quantiles(strata_values[name]),
                                                  "above_threshold": strata_above[name], "above_threshold_rate": strata_above[name] / max(1, strata_capacity[name])}
                                          for name in STRATA}},
        "fixed_threshold_metrics": {"accepted": totals["accepted"], "true_positives": totals["tp"], "false_positives": totals["fp"],
                                     "false_negatives": totals["fn"], "precision": totals["tp"] / max(1, total_accepted),
                                     "recall": totals["tp"] / max(1, totals["truth"]), "prohibited_structure_hits": totals["prohibited"],
                                     "accepted_false_positive_attribution": dict(accepted_fp)},
        "truth_mask_scenarios": scenario,
        "diagnostic_conclusion": "fixed-threshold failure concentration is reported by per-stratum above-threshold rates; no threshold change is proposed",
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = summarize(args.model.resolve()); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
