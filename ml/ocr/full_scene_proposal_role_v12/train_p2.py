# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Single-use zero-optimizer P2 parity repair for OCR V12."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from ml.markers.gate_seal import canonical_json_bytes, sha256_file, source_bundle_sha256
from ml.markers.training_budget import acquire_training_candidate, complete_training_candidate
from .dataset import build_split, encode_proposal, proposals, proposal_summary, split_fingerprint
from .model import FullSceneProposalRoleNet
from .pipeline import evaluate_thresholds
from .protocol import ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR, REVISION, ROLE_ACCURACY_MINIMUM, ROLE_CLASS_ACCURACY_MINIMUM, TASK


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/full_scene_proposal_role_v12")
CANDIDATE_ID = "P2"
CONFIG_PATH = ROOT / "training/p2.json"
SELECTION_PATH = ROOT / "SELECTION_MANIFEST.json"
SEAL_PATH = ROOT / "SEALED_PUBLIC_TEST_SEAL.json"
P1_RESULT_PATH = ROOT / "P1_RESULT.json"
CANONICAL_OUTPUT = ROOT / "artifacts/P2-run"
RUNNER_SOURCE_PATHS = (
    ROOT / "dataset.py", ROOT / "model.py", ROOT / "pipeline.py", ROOT / "protocol.py", ROOT / "train_p2.py",
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/protocol.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
    Path("ml/markers/gate_seal.py"), Path("ml/markers/training_budget.py"),
)


class ScaledCandidate(nn.Module):
    def __init__(self, candidate: FullSceneProposalRoleNet, scale: float) -> None:
        super().__init__()
        self.candidate = candidate
        self.scale = scale

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.candidate(value) * self.scale


def _export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    torch.onnx.export(
        model, example, path, input_names=["region_proposals"], output_names=["proposal_role_logits"],
        dynamic_axes={"region_proposals": {0: "proposal_count"}, "proposal_role_logits": {0: "proposal_count"}},
        opset_version=18, dynamo=False,
    )


def run_candidate(output_dir: Path) -> dict[str, object]:
    if output_dir.exists():
        raise RuntimeError(f"OCR V12 P2 output exists: {output_dir}")
    config = json.loads((REPO_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    authorization = acquire_training_candidate(
        REPO_ROOT, task=TASK, revision=REVISION, candidate_id=CANDIDATE_ID,
        config_path=CONFIG_PATH, runner_source_paths=RUNNER_SOURCE_PATHS,
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / "candidate-report.json"
    started, phase = time.perf_counter(), "initialization"
    try:
        if config["expected_runner_source_bundle_sha256"] != source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS):
            raise RuntimeError("OCR V12 P2 runner sources changed")
        if config["selection_manifest_sha256"] != sha256_file(REPO_ROOT / SELECTION_PATH):
            raise RuntimeError("OCR V12 P2 selection manifest changed")
        if config["sealed_public_test_seal_sha256"] != sha256_file(REPO_ROOT / SEAL_PATH):
            raise RuntimeError("OCR V12 P2 public seal changed")
        if config["p1_result_sha256"] != sha256_file(REPO_ROOT / P1_RESULT_PATH):
            raise RuntimeError("OCR V12 P1 result changed")
        p1 = json.loads((REPO_ROOT / P1_RESULT_PATH).read_text(encoding="utf-8"))
        checkpoint_path = REPO_ROOT / "ml/ocr/full_scene_proposal_role_v12/artifacts/P1-run/graph-text-full-scene-proposal-role-v12-p1.pt"
        if p1["checkpoint_sha256"] != config["p1_checkpoint_sha256"] or sha256_file(checkpoint_path) != config["p1_checkpoint_sha256"]:
            raise RuntimeError("OCR V12 P1 checkpoint changed")
        if p1["report_sha256"] != config["p1_report_sha256"]:
            raise RuntimeError("OCR V12 P1 report binding changed")
        validation = build_split("validation")
        selection = json.loads((REPO_ROOT / SELECTION_PATH).read_text(encoding="utf-8"))
        if split_fingerprint(validation) != selection["validation"]["split_fingerprint"]:
            raise RuntimeError("OCR V12 P2 validation split changed")
        if proposal_summary(validation) != {key: selection["validation"][key] for key in proposal_summary(validation)}:
            raise RuntimeError("OCR V12 P2 validation proposals changed")
        base = FullSceneProposalRoleNet(seed=int(config["seed"]))
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        base.load_state_dict(state["state_dict"], strict=True)
        model = ScaledCandidate(base, float(config["output_scale"])).eval()
        parity_arrays: list[np.ndarray] = []
        for scene in validation:
            items = proposals(scene.raster)
            parity_arrays.extend(encode_proposal(scene.raster, item) for item in items)
            if len(parity_arrays) >= 256:
                break
        parity_values = torch.from_numpy(np.stack(parity_arrays[:256]).astype(np.float32))
        phase = "export"
        onnx_path = output_dir / "graph-text-full-scene-proposal-role-v12-p2.onnx"
        _export(model, parity_values, onnx_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        if session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("OCR V12 P2 requires CPU selection execution")
        with torch.inference_mode():
            expected = model(parity_values).numpy()
        actual = np.asarray(session.run(None, {"region_proposals": parity_values.numpy()})[0], dtype=np.float32)
        maximum_error = float(np.max(np.abs(expected - actual)))
        parity_passed = maximum_error <= ONNX_PARITY_MAXIMUM_ABSOLUTE_ERROR
        input_digest, output_digest, calls = sha256(), sha256(), 0

        def runner(input_values: np.ndarray) -> np.ndarray:
            nonlocal calls
            contiguous = np.ascontiguousarray(input_values, dtype=np.float32)
            input_digest.update(contiguous.tobytes(order="C"))
            output = np.asarray(session.run(None, {"region_proposals": contiguous})[0], dtype=np.float32)
            output_digest.update(np.ascontiguousarray(output).tobytes(order="C"))
            calls += 1
            return output

        phase = "selection"
        comparisons = evaluate_thresholds(validation, runner, tuple(float(value) for value in config["selection_thresholds"]))
        selected = max(comparisons, key=lambda item: (
            item["metrics"]["exact_scene_count"], -item["metrics"]["false_positives"],
            -item["metrics"]["false_negatives"], item["metrics"]["role_accuracy"], item["threshold"],
        ))
        metrics = selected["metrics"]
        passed = (
            metrics["exact_scene_count"] == metrics["scene_count"]
            and metrics["true_positives"] == metrics["truth_region_count"]
            and metrics["false_positives"] == metrics["false_negatives"] == metrics["duplicate_region_count"] == metrics["prohibited_structure_hits"] == 0
            and metrics["role_accuracy"] >= ROLE_ACCURACY_MINIMUM
            and min(metrics["per_role_accuracy"].values()) >= ROLE_CLASS_ACCURACY_MINIMUM
            and parity_passed and calls == len(validation)
        )
        report: dict[str, object] = {
            "schema": "graphreader.ocr-full-scene-proposal-role-candidate.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID,
            "status": "selected" if passed else "failed_selection", "selection_gate_passed": passed,
            "production_approval": False, "release_eligible": False, "optimizer_steps": 0,
            "weights_changed": False, "p1_checkpoint_sha256": config["p1_checkpoint_sha256"],
            "output_scale": config["output_scale"], "onnx_path": onnx_path.relative_to(REPO_ROOT).as_posix(),
            "onnx_sha256": sha256_file(onnx_path), "onnx_parity_maximum_absolute_error": maximum_error,
            "onnx_parity_passed": parity_passed, "provider": "CPUExecutionProvider",
            "threshold_comparisons": comparisons, "selected_threshold": selected["threshold"],
            "selection_metrics": metrics, "direct_execution": {
                "inference_calls": calls, "input_tensor_stream_sha256": input_digest.hexdigest(),
                "output_tensor_stream_sha256": output_digest.hexdigest(),
            },
            "sealed_public_archive_opened": False, "public_gate_evaluations": 0,
            "training_authorization": authorization.binding,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        report_path.write_bytes(canonical_json_bytes(report))
        complete_training_candidate(authorization, status=str(report["status"]), report_sha256=sha256_file(report_path))
        return report
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-full-scene-proposal-role-failure.v1", "task": TASK,
            "revision": REVISION, "candidate_id": CANDIDATE_ID, "status": "failed_runner",
            "phase": phase, "optimizer_steps": 0, "exception_type": type(error).__name__,
            "exception_message": str(error), "completed_utc": datetime.now(timezone.utc).isoformat(),
            "production_approval": False, "release_eligible": False, "public_gate_evaluations": 0,
            "sealed_public_archive_opened": False, "training_authorization": authorization.binding,
        }
        report_path.write_bytes(canonical_json_bytes(failure))
        complete_training_candidate(authorization, status="failed_runner", report_sha256=sha256_file(report_path))
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CANONICAL_OUTPUT)
    args = parser.parse_args()
    report = run_candidate(REPO_ROOT / args.output)
    print(json.dumps({
        "status": report["status"], "optimizer_steps": report["optimizer_steps"],
        "selected_threshold": report["selected_threshold"], "selection_metrics": report["selection_metrics"],
        "onnx_parity_maximum_absolute_error": report["onnx_parity_maximum_absolute_error"],
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


if __name__ == "__main__":
    raise SystemExit(main())
