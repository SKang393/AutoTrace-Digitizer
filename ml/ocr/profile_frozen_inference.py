# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Authorization-safe timing profile for the frozen V30 OCR evidence path."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.ocr.official_bakeoff.production_evaluate import read_character_alphabet
from ml.ocr.unanimous_structure_veto_v30 import dataset
from ml.ocr.unanimous_structure_veto_v30.model import UnanimousStructureVetoProposalNet
from ml.ocr.unanimous_structure_veto_v30.pipeline import extract_relational_evidence
from ml.ocr.unanimous_structure_veto_v30.train_p1 import _unanimous_objective
from ml.ocr.unanimous_structure_veto_v30.protocol import (
    DETECTOR_PATH,
    DETECTOR_SHA256,
    FEATURE_COUNT,
    RECOGNIZER_PATH,
    RECOGNIZER_SHA256,
    RECOGNIZER_YAML_PATH,
    RECOGNIZER_YAML_SHA256,
    RELATION_FEATURE_COUNT,
    ROLE_PARENT_CHECKPOINT_PATH,
    ROLE_PARENT_CHECKPOINT_SHA256,
    SEED,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_PATH = REPO_ROOT / "ml/ocr/unanimous_structure_veto_v30/training/p1.json"
ROLE_PARENT_PATH = REPO_ROOT / ROLE_PARENT_CHECKPOINT_PATH
ALLOWED_SPLITS = frozenset(("train", "validation"))
PROFILE_SAFETY = {
    "sealed_reads": 0,
    "candidate_acquisitions": 0,
    "private_data": False,
    "model_revision_opened": False,
    "checkpoints_written": 0,
    "profile_output_files_written": 0,
}


@dataclass
class Timing:
    detector_inference_ms: float = 0.0
    recognizer_inference_ms: float = 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_frozen_inputs() -> dict[str, Any]:
    inputs = (
        ("detector", DETECTOR_PATH, DETECTOR_SHA256),
        ("recognizer", RECOGNIZER_PATH, RECOGNIZER_SHA256),
        ("recognizer_inference_yaml", RECOGNIZER_YAML_PATH, RECOGNIZER_YAML_SHA256),
        ("role_parent_checkpoint", ROLE_PARENT_CHECKPOINT_PATH, ROLE_PARENT_CHECKPOINT_SHA256),
    )
    identity: dict[str, Any] = {"provider": "CPUExecutionProvider"}
    for name, relative_path, expected in inputs:
        path = REPO_ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"missing frozen {name}: {relative_path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen {name} checksum mismatch")
        identity[f"{name}_path"] = relative_path
        identity[f"{name}_sha256"] = actual
        identity[f"{name}_expected_sha256"] = expected
    return identity


def validate_profile_scope(split: str) -> None:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"profile accepts only train or validation, got {split!r}")


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.use_deterministic_compute = True
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError("frozen profile requires CPUExecutionProvider only")
    return session


def _render_subset(split: str, scene_count: int) -> tuple[Any, ...]:
    validate_profile_scope(split)
    if scene_count < 1:
        raise ValueError("scene_count must be positive")
    return tuple(dataset.render_scene(split, index) for index in range(scene_count))


def _balanced_weights(targets: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=2).to(torch.float32)
    if torch.any(counts == 0):
        raise RuntimeError("profile train subset lacks both proposal classes")
    weights = counts.reciprocal()
    return weights * (2.0 / weights.sum())


def _optimizer_profile(
    values: np.ndarray,
    crops: np.ndarray,
    labels: np.ndarray,
    relations: tuple[np.ndarray, ...],
    scene_slices: tuple[slice, ...],
    config: dict[str, Any],
) -> tuple[float, int, int]:
    chosen = next(
        (index for index, scene_slice in enumerate(scene_slices)
         if np.unique(labels[scene_slice]).size == 2),
        None,
    )
    if chosen is None:
        raise RuntimeError("profile train subset has no balanced scene")
    model = UnanimousStructureVetoProposalNet(seed=int(config["seed"]))
    payload = torch.load(ROLE_PARENT_PATH, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("profile role-parent state is missing")
    model.load_role_parent_state_dict(state)
    trainable = model.trainable_parameters()
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scene_slice = scene_slices[chosen]
    evidence = torch.from_numpy(values[scene_slice]).unsqueeze(0)
    scene_crops = torch.from_numpy(crops[scene_slice]).unsqueeze(0)
    scene_relations = torch.from_numpy(relations[chosen]).unsqueeze(0)
    targets = torch.from_numpy(labels[scene_slice])
    class_weights = _balanced_weights(targets)
    model.train()
    started = time.perf_counter()
    consensus, attention, summary, local_veto = model.proposal_routes(
        evidence, scene_crops, scene_relations,
    )
    loss, _ = _unanimous_objective(
        consensus[0], attention[0], summary[0], local_veto[0],
        targets, class_weights, config,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(
        trainable, max_norm=float(config["gradient_clip_norm"]),
    )
    optimizer.step()
    del attention, summary, local_veto
    return (time.perf_counter() - started) * 1000.0, 1, int(chosen)


def profile(train_scenes: int = 2, validation_scenes: int = 2) -> dict[str, Any]:
    if train_scenes < 1 or validation_scenes < 1:
        raise ValueError("both scene counts must be positive")
    frozen_inputs = validate_frozen_inputs()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    render_started = time.perf_counter()
    train = _render_subset("train", train_scenes)
    validation = _render_subset("validation", validation_scenes)
    render_ms = (time.perf_counter() - render_started) * 1000.0

    detector_path = REPO_ROOT / DETECTOR_PATH
    recognizer_path = REPO_ROOT / RECOGNIZER_PATH
    detector_init_started = time.perf_counter()
    detector_session = _session(detector_path)
    detector_init_ms = (time.perf_counter() - detector_init_started) * 1000.0
    recognizer_init_started = time.perf_counter()
    recognizer_session = _session(recognizer_path)
    recognizer_init_ms = (time.perf_counter() - recognizer_init_started) * 1000.0
    detector_input = detector_session.get_inputs()[0].name
    recognizer_input = recognizer_session.get_inputs()[0].name
    timing = Timing()

    def detector_runner(values: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        output = detector_session.run(None, {
            detector_input: np.ascontiguousarray(values),
        })[0]
        timing.detector_inference_ms += (time.perf_counter() - started) * 1000.0
        return np.asarray(output, dtype=np.float32)

    def recognizer_runner(values: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        output = recognizer_session.run(None, {
            recognizer_input: np.ascontiguousarray(values),
        })[0]
        timing.recognizer_inference_ms += (time.perf_counter() - started) * 1000.0
        return np.asarray(output, dtype=np.float32)

    config = json.loads(TRAINING_CONFIG_PATH.read_text(encoding="utf-8"))
    alphabet = read_character_alphabet(REPO_ROOT / RECOGNIZER_YAML_PATH)
    extraction: dict[str, dict[str, Any]] = {}
    train_payload: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, ...], tuple[slice, ...]] | None = None
    for split, scenes, mode in (("train", train, "train"), ("validation", validation, "train")):
        before_detector = timing.detector_inference_ms
        before_recognizer = timing.recognizer_inference_ms
        started = time.perf_counter()
        values, crops, labels, _, relations, slices, evidence = extract_relational_evidence(
            scenes,
            detector_runner,
            recognizer_runner,
            alphabet,
            mode=mode,
            negative_cap_per_scene=int(config["complete_proposal_negative_cap_per_scene"]),
            recognition_batch_size=int(config["recognition_batch_size"]),
        )
        pipeline_ms = (time.perf_counter() - started) * 1000.0
        detector_ms = timing.detector_inference_ms - before_detector
        recognizer_ms = timing.recognizer_inference_ms - before_recognizer
        extraction[split] = {
            "scene_count": len(scenes),
            "proposal_count": int(len(values)),
            "detector_inference_calls": evidence["detector_inference_calls"],
            "recognizer_batch_calls": evidence["recognizer_batch_calls"],
            "pipeline_total_ms": round(pipeline_ms, 3),
            "detector_inference_ms": round(detector_ms, 3),
            "recognizer_inference_ms": round(recognizer_ms, 3),
            "feature_crop_relation_preparation_ms": round(
                max(0.0, pipeline_ms - detector_ms - recognizer_ms), 3,
            ),
            "feature_label_stream_sha256": evidence["feature_label_stream_sha256"],
        }
        if split == "train":
            train_payload = (values, crops, relations, slices)
            train_labels = labels

    if train_payload is None:
        raise RuntimeError("train profile payload was not produced")
    optimizer_ms, optimizer_steps, optimizer_scene = _optimizer_profile(
        train_payload[0], train_payload[1], train_labels,
        train_payload[2], train_payload[3], config,
    )
    report = {
        "schema": "graphreader.ocr-frozen-inference-profile.v1",
        "revision_under_profile": "graph-text-unanimous-structure-veto-v30",
        "scene_scope": {"train": train_scenes, "validation": validation_scenes},
        "frozen_inputs": frozen_inputs,
        "timing_ms": {
            "procedural_data_preparation": round(render_ms, 3),
            "detector_session_init": round(detector_init_ms, 3),
            "recognizer_session_init": round(recognizer_init_ms, 3),
            "frozen_detector_inference": round(timing.detector_inference_ms, 3),
            "frozen_recognizer_inference": round(timing.recognizer_inference_ms, 3),
            "feature_crop_relation_preparation": round(sum(
                float(item["feature_crop_relation_preparation_ms"])
                for item in extraction.values()
            ), 3),
            "bounded_optimizer_pass": round(optimizer_ms, 3),
        },
        "splits": extraction,
        "optimizer": {
            "steps": optimizer_steps,
            "scene_index": optimizer_scene,
            "checkpoint_written": False,
        },
        "safety": dict(PROFILE_SAFETY),
        "safety_definition": {
            "model_revision_opened": (
                "no candidate authorization or revision state was opened; frozen "
                "model bytes were checksum-verified and loaded read-only for timing"
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scenes", type=int, default=2)
    parser.add_argument("--validation-scenes", type=int, default=2)
    args = parser.parse_args()
    report = profile(args.train_scenes, args.validation_scenes)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
