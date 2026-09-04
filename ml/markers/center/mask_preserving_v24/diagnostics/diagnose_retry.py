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
from ml.markers.center.real_range_generator_v1.negative_sampler import (
    CONNECTOR_ANCHOR_FRACTIONS,
    CONNECTOR_ANCHOR_MAX_DISTANCE_PX,
    CONNECTOR_ENDPOINT_OFFSET_PX,
    TOPOLOGY_KINDS,
    TOPOLOGY_RADIUS_PX,
    TOPOLOGY_SAMPLER_RADIUS_PX,
    _connector_anchor_indices,
    _topology_indices,
)
from ..mask_preserving import extract_proposals, postprocess, prohibited_hits

ROOT = Path(__file__).resolve().parents[5]
THRESHOLD = 0.25
TOLERANCE = 5.0
STRATA = ("hard_existing", "faint_low", "faint_p05", "ocr_heavy", "artifact", "generic")
MORPHOLOGY_KEYS = ("dark_fraction_ge_012", "dark_fraction_ge_05", "center5x5_mean", "max_row_dark_fraction_ge_012", "max_col_dark_fraction_ge_012", "max_row_dark_fraction_ge_05", "max_col_dark_fraction_ge_05", "foreground_extent_balance", "covariance_eigen_ratio", "border_dark_fraction_ge_012", "max_ring_support_3_12")
RETRY2_ONNX_SHA256 = "98c605eea8a579d28ad0e5d3b355458ab1e1883c3947c4a3a442557d639f5b79"
RETRY3_ONNX_SHA256 = "0d80d1994d7b33241c795c9e6f92c802750555a62c3cd3335777eb969fb5083a"
RETRY3_GENERATOR_AUDIT_SHA256 = "3568fff359e3541be14bf1f02774887c8110ded9f02fd95a8f4ff680e8639d69"
RETRY3_DEV_SPLIT_SHA256 = "050df194849c9e787d786624b26fd268e7f7a1832c271868521d89bf6588e960"
RETRY4_ONNX_SHA256 = "697fbcfb961e4c2af36a1a3d68cf5be874412b2939b03c42b59aaa82c4b0de96"
RETRY4_GENERATOR_AUDIT_SHA256 = "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
RETRY4_DEV_SPLIT_SHA256 = "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"
RETRY5_ONNX_SHA256 = "d3445f0b1bf0e97a98942133d45341cae75548887be853743e887832cacad7bd"
RETRY5_GENERATOR_AUDIT_SHA256 = "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
RETRY5_DEV_SPLIT_SHA256 = "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"
RETRY6_ONNX_SHA256 = "31d473d6c24bf21edc1cbfb25f7da35eabfed7cbf8afc13bf52bef23d06bfeb9"
RETRY6_GENERATOR_AUDIT_SHA256 = "9562044526bce45b48254472346ccc1640c6e254915b9e744525154144121748"
RETRY6_DEV_SPLIT_SHA256 = "f453e50e228f54e15bafba83b1a5dda422c435a555e9083dfea18457ed38204d"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    q = np.quantile(np.asarray(values, dtype=np.float64), [0, .05, .50, .90, .95, 1])
    return {"count": len(values), "minimum": float(q[0]), "p05": float(q[1]),
            "median": float(q[2]), "p90": float(q[3]), "p95": float(q[4]),
            "maximum": float(q[5])}

def _patch_morphology(patch: torch.Tensor) -> dict[str, float]:
    ink = patch[0].detach().cpu().numpy().astype(np.float64, copy=False); d12, d05 = ink >= .12, ink >= .5; center = ink[14:19, 14:19]; ys, xs = np.where(d12)
    if len(xs): width, height = xs.max()-xs.min()+1, ys.max()-ys.min()+1; extent = min(width,height)/max(width,height)
    else: extent = 0.0
    if len(xs) >= 2:
        eigen = np.linalg.eigvalsh(np.cov(np.stack((xs,ys)), bias=True)); ratio = float(np.clip(eigen[1]/max(eigen[0], 1e-12), 1.0, 1e6))
    else: ratio = 1.0
    border = np.concatenate((d12[0], d12[-1], d12[1:-1,0], d12[1:-1,-1])); ring = []
    for radius in range(3,13):
        points = tuple((int(round(16 + radius * np.cos(i*np.pi/4))), int(round(16 + radius * np.sin(i*np.pi/4)))) for i in range(8)); ring.append(sum(0 <= x < 33 and 0 <= y < 33 and ink[y,x] >= .12 for x,y in points))
    return {"dark_fraction_ge_012":float(d12.mean()),"dark_fraction_ge_05":float(d05.mean()),"center5x5_mean":float(center.mean()),"max_row_dark_fraction_ge_012":float(d12.mean(1).max()),"max_col_dark_fraction_ge_012":float(d12.mean(0).max()),"max_row_dark_fraction_ge_05":float(d05.mean(1).max()),"max_col_dark_fraction_ge_05":float(d05.mean(0).max()),"foreground_extent_balance":float(extent),"covariance_eigen_ratio":ratio,"border_dark_fraction_ge_012":float(border.mean()),"max_ring_support_3_12":float(max(ring,default=0))}


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


def summarize(model_path: Path, *, retry4: bool = False, retry5: bool = False, retry6: bool = False) -> dict:
    started = time.perf_counter()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model_hash = _sha(model_path)
    if sum((retry4, retry5, retry6)) > 1:
        raise ValueError("retry modes are mutually exclusive")
    expected_model = RETRY6_ONNX_SHA256 if retry6 else RETRY5_ONNX_SHA256 if retry5 else RETRY4_ONNX_SHA256 if retry4 else "aee3b4ba47197d84eeecacc631730e67fd99c261b913643f95c25a8ea2436c11"
    if model_hash != expected_model:
        mode = "retry6" if retry6 else "retry5" if retry5 else "retry4" if retry4 else "retry1"
        raise ValueError(f"{mode} ONNX hash mismatch: expected {expected_model}, got {model_hash}")
    sampler_path = ROOT / "ml/markers/center/real_range_generator_v1/negative_sampler.py"
    audit_path = ROOT / "ml/markers/center/real_range_generator_v1/AUDIT.json"
    config_path = ROOT / "ml/markers/center/mask_preserving_v24/training/p1.json"
    audit_hash = _sha(audit_path)
    audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_dev_split_sha256 = audit_record["splits"]["dev"]["aggregate_sha256"]
    expected_audit = RETRY6_GENERATOR_AUDIT_SHA256 if retry6 else RETRY5_GENERATOR_AUDIT_SHA256 if retry5 else RETRY4_GENERATOR_AUDIT_SHA256
    expected_dev = RETRY6_DEV_SPLIT_SHA256 if retry6 else RETRY5_DEV_SPLIT_SHA256 if retry5 else RETRY4_DEV_SPLIT_SHA256
    if (retry4 or retry5 or retry6) and audit_hash != expected_audit:
        raise ValueError(f"retry audit hash mismatch: expected {expected_audit}, got {audit_hash}")
    if (retry4 or retry5 or retry6) and actual_dev_split_sha256 != expected_dev:
        raise ValueError(f"retry dev split hash mismatch: expected {expected_dev}, got {actual_dev_split_sha256}")
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
    topology_capacity = {kind: 0 for kind in TOPOLOGY_KINDS}
    topology_above = {kind: 0 for kind in TOPOLOGY_KINDS}
    connector_capacity = connector_above = 0
    retry5_above = defaultdict(int)
    retry5_fp = defaultdict(int)
    retry6_above = defaultdict(int)
    retry6_fp = defaultdict(int)
    prohibited_by_kind = defaultdict(int)
    prohibited_by_source = defaultdict(int)
    prohibited_confidence = []
    prohibited_distance = []
    prohibited_ocr = []
    prohibited_artifact = []
    prohibited_morphology = defaultdict(list)
    for scene in scenes:
        proposals = extract_proposals(scene.tensor)
        labels, hard = _labels(scene, proposals.coordinates)
        names = _strata(proposals.patches, hard, labels)
        topology = _topology_indices(scene, proposals.coordinates, labels) if (retry4 or retry5 or retry6) else {kind: set() for kind in TOPOLOGY_KINDS}
        topology_by_index = {index: kind for kind in TOPOLOGY_KINDS for index in topology[kind]}
        connector_indices = _connector_anchor_indices(scene, proposals.coordinates, labels) if (retry5 or retry6) else set()
        connector_indices -= set(topology_by_index)
        output = session.run([output_name], {input_name: proposals.patches.numpy().astype(np.float32, copy=False)})[0]
        probabilities = output[:, 0].astype(np.float64)
        if retry4 or retry5 or retry6:
            for kind in TOPOLOGY_KINDS:
                topology_capacity[kind] += len(topology[kind])
                topology_above[kind] += sum(int(probabilities[index] >= THRESHOLD) for index in topology[kind])
        if retry5 or retry6:
            connector_capacity += len(connector_indices)
            connector_above += sum(int(probabilities[index] >= THRESHOLD) for index in connector_indices)
        for i, name in enumerate(names):
            if name is None:
                positives.append(float(probabilities[i]))
            else:
                strata_capacity[name] += 1; strata_values[name].append(float(probabilities[i]))
                strata_above[name] += int(probabilities[i] >= THRESHOLD)
                if retry5 and probabilities[i] >= THRESHOLD:
                    retry5_above[topology_by_index.get(i, "connector_anchor" if i in connector_indices else name)] += 1
                if retry6 and probabilities[i] >= THRESHOLD:
                    retry6_above[topology_by_index.get(i, "connector_anchor" if i in connector_indices else name)] += 1
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
                name = topology_by_index.get(source, "connector_anchor" if (retry5 or retry6) and source in connector_indices else names[source]) if (retry4 or retry5 or retry6) else names[source]
                accepted_fp[name or "unattributed"] += 1
                if retry5:
                    retry5_fp[name or "unattributed"] += 1
                if retry6:
                    retry6_fp[name or "unattributed"] += 1
            else:
                accepted_fp["unattributed"] += 1
        for local, center in enumerate(scene.centers):
            x, y = (int(round(value)) for value in center)
            ocr = float(scene.tensor[1, max(0, y-2):y+3, max(0, x-2):x+3].max()) >= .35
            artifact = float(scene.tensor[2, max(0, y-2):y+3, max(0, x-2):x+3].max()) >= .35
            key = "both_masks" if ocr and artifact else "ocr_mask" if ocr else "artifact_mask" if artifact else "unmasked"
            scenario[key]["truth"] += 1; scenario[key]["tp"] += int(local in used_t); scenario[key]["fn"] += int(local not in used_t)
        if retry6:
            for prediction in predictions:
                source = min(decoded, key=lambda item: math.hypot(prediction.x-item[0], prediction.y-item[1]))[2] if decoded else None
                if source is None:
                    continue
                patch = proposals.patches[source]
                source_name = topology_by_index.get(source, "connector_anchor" if source in connector_indices else names[source])
                for kind, x, y in scene.hard_negatives:
                    distance = math.hypot(prediction.x - x, prediction.y - y)
                    if distance > TOLERANCE:
                        continue
                    prohibited_by_kind[kind] += 1
                    prohibited_by_source[source_name or "unattributed"] += 1
                    prohibited_confidence.append(float(output[source, 0]))
                    prohibited_distance.append(distance)
                    prohibited_ocr.append(float(patch[1].mean()))
                    prohibited_artifact.append(float(patch[2].mean()))
                    for key, value in _patch_morphology(patch).items():
                        prohibited_morphology[key].append(float(value))
    for value in scenario.values():
        value["recall"] = value["tp"] / max(1, value["truth"])
    total_accepted = totals["tp"] + totals["fp"]
    report = {
        "schema": "graphreader.marker-center-mask-preserving-v24-retry6-diagnosis.v1" if retry6 else "graphreader.marker-center-mask-preserving-v24-retry5-diagnosis.v1" if retry5 else "graphreader.marker-center-mask-preserving-v24-retry4-diagnosis.v1" if retry4 else "graphreader.marker-center-mask-preserving-v24-retry-diagnosis.v1",
        "revision": "marker-center-mask-preserving-v24",
        "scope": {"synthetic_only": True, "split": "real-range-generator-v1-dev", "scene_count": len(scenes),
                  "truth_count": totals["truth"], "label_positive_distance_px": 3.0, "threshold": THRESHOLD,
                  "private_data": False, "real_dev_reads": 0, "real_sealed_reads": 0, "optimizer_steps": 0,
                  "case_ids_or_pixels_emitted": False, "retry_mode": "retry6" if retry6 else "retry5" if retry5 else "retry4" if retry4 else "retry1"},
        "binding": {"model_path": model_path.name, "model_sha256": model_hash, "provider": session.get_providers()[0],
                    "input_shape": ["candidate_count", 3, 33, 33], "output_shape": ["candidate_count", 4],
                    "generator_audit_sha256": audit_hash, "generator_dev_split_sha256": actual_dev_split_sha256 if (retry4 or retry5 or retry6) else json.loads(config_path.read_text())["dev_split_sha256"],
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
    if retry4 or retry5 or retry6:
        report["topology"] = {"radius_px": TOPOLOGY_RADIUS_PX, "capacity": topology_capacity, "above_threshold": topology_above, "accepted_false_positive_attribution": {kind: int(accepted_fp.get(kind, 0)) for kind in TOPOLOGY_KINDS}}
    if retry5:
        report["retry5_attribution"] = {
            "above_threshold": {**{kind: int(topology_above[kind]) for kind in TOPOLOGY_KINDS}, "connector_anchor": int(connector_above), **{name: int(value) for name, value in retry5_above.items() if name not in TOPOLOGY_KINDS and name != "connector_anchor"}},
            "accepted_false_positives": {**{kind: int(accepted_fp.get(kind, 0)) for kind in TOPOLOGY_KINDS}, "connector_anchor": int(retry5_fp.get("connector_anchor", 0)), **{name: int(value) for name, value in retry5_fp.items() if name not in TOPOLOGY_KINDS and name != "connector_anchor"}},
            "connector_anchor_fractions": list(CONNECTOR_ANCHOR_FRACTIONS),
            "connector_anchor_max_distance_px": CONNECTOR_ANCHOR_MAX_DISTANCE_PX,
            "connector_anchor_capacity": int(connector_capacity),
        }
    if retry6:
        report["retry6_attribution"] = {
            "above_threshold": {**{kind: int(topology_above[kind]) for kind in TOPOLOGY_KINDS}, "connector_anchor": int(connector_above), **{name: int(value) for name, value in retry6_above.items() if name not in TOPOLOGY_KINDS and name != "connector_anchor"}},
            "accepted_false_positives": {**{kind: int(accepted_fp.get(kind, 0)) for kind in TOPOLOGY_KINDS}, "connector_anchor": int(retry6_fp.get("connector_anchor", 0)), **{name: int(value) for name, value in retry6_fp.items() if name not in TOPOLOGY_KINDS and name != "connector_anchor"}},
            "topology_capacity": {kind: int(topology_capacity[kind]) for kind in TOPOLOGY_KINDS},
            "connector_anchor_capacity": int(connector_capacity),
            "topology_sampler_radius_px": TOPOLOGY_SAMPLER_RADIUS_PX,
            "connector_endpoint_offset_px": CONNECTOR_ENDPOINT_OFFSET_PX,
            "connector_anchor_max_distance_px": CONNECTOR_ANCHOR_MAX_DISTANCE_PX,
        }
        report["prohibited_hit_attribution"] = {
            "total": int(sum(prohibited_by_kind.values())),
            "by_prohibited_kind": dict(sorted(prohibited_by_kind.items())),
            "by_source_group": dict(sorted(prohibited_by_source.items())),
            "confidence": _quantiles(prohibited_confidence),
            "distance_px": _quantiles(prohibited_distance),
            "ocr_mean": _quantiles(prohibited_ocr),
            "artifact_mean": _quantiles(prohibited_artifact),
            "morphology": {key: _quantiles(values) for key, values in sorted(prohibited_morphology.items())},
        }
    return report


def summarize_morphology(model_path: Path, *, retry3: bool = False) -> dict:
    started = time.perf_counter()
    expected = RETRY3_ONNX_SHA256 if retry3 else RETRY2_ONNX_SHA256
    model_hash = _sha(model_path)
    if model_hash != expected:
        mode = "retry3" if retry3 else "retry2"
        raise ValueError(f"{mode} ONNX hash mismatch: expected {expected}, got {model_hash}")
    config_path=ROOT/"ml/markers/center/mask_preserving_v24/training/p1.json"; sampler_path=ROOT/"ml/markers/center/real_range_generator_v1/negative_sampler.py"; audit_path=ROOT/"ml/markers/center/real_range_generator_v1/AUDIT.json"
    audit_hash = _sha(audit_path)
    if retry3 and audit_hash != RETRY3_GENERATOR_AUDIT_SHA256:
        raise ValueError(f"retry3 generator audit hash mismatch: expected {RETRY3_GENERATOR_AUDIT_SHA256}, got {audit_hash}")
    audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
    actual_dev_split_sha256 = audit_record["splits"]["dev"]["aggregate_sha256"]
    if retry3 and actual_dev_split_sha256 != RETRY3_DEV_SPLIT_SHA256:
        raise ValueError(f"retry3 dev split hash mismatch: expected {RETRY3_DEV_SPLIT_SHA256}, got {actual_dev_split_sha256}")
    session=ort.InferenceSession(str(model_path),providers=["CPUExecutionProvider"])
    if session.get_providers()[0] != "CPUExecutionProvider": raise RuntimeError("CPUExecutionProvider was not selected")
    inp,out=session.get_inputs()[0].name,session.get_outputs()[0].name
    buckets={name:{key:[] for key in MORPHOLOGY_KEYS} for name in ("positives","negative_below_025","negative_above_025","accepted_generic_false_positives")}; capacities=defaultdict(int); above=defaultdict(int)
    scenes = build_split("dev")
    for scene in scenes:
        proposals=extract_proposals(scene.tensor); labels,hard=_labels(scene,proposals.coordinates); names=_strata(proposals.patches,hard,labels); output=session.run([out],{inp:proposals.patches.numpy().astype(np.float32,copy=False)})[0]
        for i,name in enumerate(names):
            bucket="positives" if name is None else "negative_above_025" if output[i,0]>=THRESHOLD else "negative_below_025"
            if name is not None: capacities[name]+=1; above[name]+=int(output[i,0]>=THRESHOLD)
            features=_patch_morphology(proposals.patches[i])
            for key in MORPHOLOGY_KEYS: buckets[bucket][key].append(features[key])
        predictions=postprocess(scene,proposals,output); used_p,_=_matched_truths(predictions,scene.centers); decoded=[(float(x+output[i,1]*4),float(y+output[i,2]*4),i) for i,(x,y) in enumerate(proposals.coordinates.tolist()) if output[i,0]>=THRESHOLD]
        for pi,prediction in enumerate(predictions):
            if pi in used_p or not decoded: continue
            source=min(decoded,key=lambda item:math.hypot(prediction.x-item[0],prediction.y-item[1]))[2]
            if names[source]=="generic":
                features=_patch_morphology(proposals.patches[source])
                for key in MORPHOLOGY_KEYS: buckets["accepted_generic_false_positives"][key].append(features[key])
    mode = "retry3" if retry3 else "retry2"
    return {"schema":f"graphreader.marker-center-mask-preserving-v24-{mode}-morphology-diagnosis.v1","revision":"marker-center-mask-preserving-v24","scope":{"synthetic_only":True,"split":"real-range-generator-v1-dev","scene_count":len(scenes),"threshold":THRESHOLD,"private_data":False,"real_dev_reads":0,"real_sealed_reads":0,"optimizer_steps":0,"case_ids_or_pixels_emitted":False,"retry_mode":mode},"binding":{"model_sha256":model_hash,"provider":session.get_providers()[0],"input_shape":["candidate_count",3,33,33],"output_shape":["candidate_count",4],"generator_audit_sha256":audit_hash,"generator_dev_split_sha256":actual_dev_split_sha256 if retry3 else json.loads(config_path.read_text())["dev_split_sha256"],"negative_sampler_sha256":_sha(sampler_path),"negative_sampler_priority":list(STRATA)},"negative_strata_counts":{name:{"capacity":capacities[name],"above_threshold":above[name],"above_threshold_rate":above[name]/max(1,capacities[name])} for name in STRATA},"morphology_quantiles":{bucket:{key:_quantiles(values) for key,values in values_by_key.items()} for bucket,values_by_key in buckets.items()},"accepted_generic_false_positive_count":len(buckets["accepted_generic_false_positives"][MORPHOLOGY_KEYS[0]]),"elapsed_ms":round((time.perf_counter()-started)*1000,3),"threshold_change_proposed":False}

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--morphology", action="store_true"); parser.add_argument("--retry3", action="store_true"); parser.add_argument("--retry4", action="store_true"); parser.add_argument("--retry5", action="store_true"); parser.add_argument("--retry6", action="store_true")
    args = parser.parse_args()
    if sum((args.retry3, args.retry4, args.retry5, args.retry6)) > 1:
        parser.error("retry modes are mutually exclusive")
    if args.retry3 and not args.morphology:
        parser.error("--retry3 requires --morphology")
    if args.retry4 and args.morphology:
        parser.error("--retry4 is a standard diagnosis mode")
    if (args.retry5 or args.retry6) and args.morphology:
        parser.error("--retry5 is a standard diagnosis mode")
    report = summarize_morphology(args.model.resolve(), retry3=args.retry3) if args.morphology else summarize(args.model.resolve(), retry4=args.retry4, retry5=args.retry5, retry6=args.retry6); args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")); print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
