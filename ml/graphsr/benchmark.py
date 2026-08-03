# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Run the deterministic, local-only GraphSR downstream benchmark.

The command never downloads weights or data.  A benchmark manifest either
points to already available local runtimes and a fixed local dataset, or the
report records each unavailable candidate as blocked/unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .metrics import METRIC_NAMES, QualityMetrics, evaluate_quality_metrics, mean_quality_metrics


BENCHMARK_SCHEMA_VERSION = 1
CONFIGURED_OUTPUT_SCALE = 2
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.001
MEASUREMENT_CONTRACT = {
    "runtime": "input decode through output RGB uint8 materialization; common shape validation and metric scoring excluded",
    "memory": "maximum combined resident working set of the benchmark host and candidate child processes sampled every 1 ms over the runtime boundary",
}
REQUIRED_CANDIDATES = (
    "RealESRGAN_x2plus",
    "realesr-general-x4v3",
    "realesr-animevideov3",
    "GraphSR-x2",
    "baseline-bicubic-x2",
)

_BASELINE_ID = "baseline-bicubic-x2"
_HIGHER_IS_BETTER = (
    "numeric_ocr_exact_match",
    "marker_center_f1",
    "shape_fill_classification_f1",
    "axis_thin_line_recall",
    "open_marker_preservation_rate",
)
_DIAGNOSTIC_METRICS = ("numeric_ocr_proxy_exact_match",)
_LOWER_IS_BETTER = (
    "marker_center_mean_error_pixels",
    "axis_line_localization_error_pixels",
    "hallucinated_structure_rate",
)
_RUNTIME_METRICS = ("runtime_ms_mean", "peak_memory_bytes")
_DEFAULT_MANIFESTS = {
    "RealESRGAN_x2plus": "models/manifest/super-resolution/RealESRGAN_x2plus.json",
    "realesr-general-x4v3": "models/manifest/super-resolution/realesr-general-x4v3-outscale2.json",
    "realesr-animevideov3": "models/manifest/super-resolution/realesr-animevideov3-ncnn-x2.json",
    "GraphSR-x2": "models/manifest/graphsr/graphsr-x2-candidate-0.1.0.json",
}


class BenchmarkManifestError(ValueError):
    """Raised when the local benchmark manifest is invalid."""


def _empty_metrics() -> dict[str, None]:
    return {name: None for name in METRIC_NAMES}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _sha256_bytes(payload)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise BenchmarkManifestError(f"Manifest is not readable: {error}") from error
    if size > 8 * 1024 * 1024:
        raise BenchmarkManifestError("Manifest exceeds the 8 MiB local limit")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkManifestError(f"Manifest is not valid UTF-8 JSON: {error}") from error
    if not isinstance(value, dict):
        raise BenchmarkManifestError("Manifest root must be a JSON object")
    return value


def _resolve_local_path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkManifestError(f"{field} must be a nonempty local path")
    lowered = value.strip().lower()
    if "://" in lowered or lowered.startswith(("data:", "file:")):
        raise BenchmarkManifestError(f"{field} must not be a URL")
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _finite_nonnegative(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value >= 0


def _valid_metrics(metrics: object) -> bool:
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        return False
    if not all(_finite_nonnegative(metrics[name]) for name in METRIC_NAMES):
        return False
    return all(
        0.0 <= float(metrics[name]) <= 1.0
        for name in (*_HIGHER_IS_BETTER, *_DIAGNOSTIC_METRICS, "hallucinated_structure_rate")
    )


def _valid_partial_metrics(metrics: object) -> bool:
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRIC_NAMES):
        return False
    for name in METRIC_NAMES:
        value = metrics[name]
        if value is None:
            continue
        if not _finite_nonnegative(value):
            return False
        if name in (*_HIGHER_IS_BETTER, *_DIAGNOSTIC_METRICS, "hallucinated_structure_rate"):
            if not 0.0 <= float(value) <= 1.0:
                return False
    return True


def _candidate_row(
    model_id: str,
    status: str,
    reason: str | None,
    *,
    runtime_kind: str,
    model_version: str | None = None,
    metrics: Mapping[str, float | int | None] | None = None,
    evidence_sha256: str | None = None,
) -> dict[str, object]:
    values = dict(metrics) if metrics is not None else _empty_metrics()
    values = {name: values.get(name) for name in METRIC_NAMES}
    if status == "blocked":
        values = _empty_metrics()
        if not reason:
            raise ValueError("Blocked or unmeasured candidates require a reason")
    elif status == "unmeasured":
        if not reason:
            raise ValueError("Blocked or unmeasured candidates require a reason")
        if not _valid_partial_metrics(values):
            raise ValueError(f"Partial metrics for {model_id} are invalid")
    elif not _valid_metrics(values):
        raise ValueError(f"Measured metrics for {model_id} are incomplete or invalid")
    return {
        "model_id": model_id,
        "model_version": model_version,
        "configured_output_scale": CONFIGURED_OUTPUT_SCALE,
        "runtime_kind": runtime_kind,
        "status": status,
        "reason": reason,
        "metrics": values,
        "evidence_sha256": evidence_sha256,
    }


def _manifest_metadata(model_id: str, repository_root: Path, config: Mapping[str, object] | None) -> tuple[str | None, str | None]:
    manifest_value = config.get("model_manifest_path") if config else None
    if manifest_value is None:
        default = _DEFAULT_MANIFESTS.get(model_id)
        manifest_path = repository_root / default if default else None
    else:
        manifest_path = _resolve_local_path(repository_root, manifest_value, "model_manifest_path")
    if manifest_path is None or not manifest_path.is_file():
        return None, None
    try:
        model_manifest = _load_json(manifest_path)
    except BenchmarkManifestError:
        return None, None
    accepted_ids = {model_id}
    if model_id == "GraphSR-x2":
        accepted_ids.add("graphsr-x2-candidate")
    if model_manifest.get("model_id") not in accepted_ids:
        return None, None
    version = model_manifest.get("model_version")
    return str(version) if isinstance(version, str) else None, _sha256_path(manifest_path)


def _normalize_case(case: object, base: Path) -> dict[str, object]:
    if not isinstance(case, Mapping):
        raise BenchmarkManifestError("Each dataset case must be an object")
    case_id = case.get("case_id")
    if not isinstance(case_id, str) or not case_id.strip():
        raise BenchmarkManifestError("Every dataset case requires a nonempty case_id")
    input_path = _resolve_local_path(base, case.get("input_path"), f"case {case_id} input_path")
    ground_truth_path = _resolve_local_path(base, case.get("ground_truth_path"), f"case {case_id} ground_truth_path")
    if not input_path.is_file() or not ground_truth_path.is_file():
        raise BenchmarkManifestError(f"Dataset files for case {case_id} are missing")
    annotations: object = case.get("annotations", {})
    if "annotations_path" in case:
        annotation_path = _resolve_local_path(base, case["annotations_path"], f"case {case_id} annotations_path")
        annotations = _load_json(annotation_path)
    if not isinstance(annotations, Mapping):
        raise BenchmarkManifestError(f"Annotations for case {case_id} must be an object")
    normalized_annotations = {
        "ocr_regions": annotations.get("ocr_regions", []),
        "marker_centers": annotations.get("marker_centers", []),
        "axis_lines": annotations.get("axis_lines", []),
        "open_markers": annotations.get("open_markers", []),
    }
    if not all(isinstance(value, list) for value in normalized_annotations.values()):
        raise BenchmarkManifestError(f"Annotation collections for case {case_id} must be arrays")
    return {
        "case_id": case_id.strip(),
        "input_path": input_path,
        "ground_truth_path": ground_truth_path,
        "annotations": normalized_annotations,
    }


def _normalize_benchmark_manifest(value: Mapping[str, object], manifest_path: Path) -> dict[str, object]:
    """Normalize a benchmark plan or an ordinary evidence-only model manifest."""

    if value.get("benchmark_manifest_version") is None:
        if value.get("manifest_version") == 1 and value.get("task") == "super_resolution":
            return {
                "dataset_id": "unconfigured",
                "cases": [],
                "candidates": {},
                "thresholds": {},
                "source_kind": "model_manifest",
            }
        raise BenchmarkManifestError("benchmark_manifest_version must equal 1")
    if value.get("benchmark_manifest_version") != BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkManifestError("Unsupported benchmark_manifest_version")
    dataset = value.get("dataset")
    if not isinstance(dataset, Mapping):
        raise BenchmarkManifestError("dataset must be an object")
    dataset_id = dataset.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise BenchmarkManifestError("dataset.dataset_id must be nonempty")
    raw_cases = dataset.get("cases")
    if not isinstance(raw_cases, list):
        raise BenchmarkManifestError("dataset.cases must be an array")
    cases = tuple(_normalize_case(case, manifest_path.parent) for case in raw_cases)
    case_ids = tuple(str(case["case_id"]) for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise BenchmarkManifestError("dataset case_id values must be unique")
    candidate_values = value.get("candidates", [])
    if not isinstance(candidate_values, list):
        raise BenchmarkManifestError("candidates must be an array")
    candidates: dict[str, Mapping[str, object]] = {}
    for candidate in candidate_values:
        if not isinstance(candidate, Mapping):
            raise BenchmarkManifestError("Each candidate must be an object")
        model_id = candidate.get("model_id")
        if model_id not in REQUIRED_CANDIDATES[:-1]:
            raise BenchmarkManifestError(f"Unknown or reserved candidate model_id: {model_id}")
        if model_id in candidates:
            raise BenchmarkManifestError(f"Duplicate candidate model_id: {model_id}")
        candidates[str(model_id)] = candidate
    selection = value.get("selection", {})
    if not isinstance(selection, Mapping):
        raise BenchmarkManifestError("selection must be an object")
    thresholds = selection.get("thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise BenchmarkManifestError("selection.thresholds must be an object")
    return {
        "dataset_id": dataset_id.strip(),
        "cases": cases,
        "candidates": candidates,
        "thresholds": dict(thresholds),
        "source_kind": "benchmark_manifest",
    }


def _load_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not decode local benchmark image {path.name}: {error}") from error


def _rss_bytes(pid: int | None = None, include_children: bool = False) -> int | None:
    try:
        import psutil

        process = psutil.Process(pid or os.getpid())
        total = int(process.memory_info().rss)
        if include_children:
            for child in process.children(recursive=True):
                try:
                    total += int(child.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        return total
    except (ImportError, OSError):
        return None


def _measure_peak_memory(operation: Any) -> tuple[Any, int | None]:
    """Sample process RSS while one in-process candidate operation runs."""

    stop = threading.Event()
    observations: list[int] = []

    def sample() -> None:
        while not stop.is_set():
            value = _rss_bytes()
            if value is not None:
                observations.append(value)
            stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)

    thread = threading.Thread(target=sample, name="graphsr-memory-sampler", daemon=True)
    thread.start()
    try:
        result = operation()
    finally:
        stop.set()
        thread.join(timeout=1.0)
        final = _rss_bytes()
        if final is not None:
            observations.append(final)
    return result, max(observations) if observations else None


def _bicubic_x2(input_path: Path) -> tuple[np.ndarray, float, int | None]:
    started = time.perf_counter()
    output, peak = _measure_peak_memory(
        lambda: _bicubic_x2_materialized(input_path)
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    return output, elapsed, peak


def _bicubic_x2_materialized(input_path: Path) -> np.ndarray:
    source = _load_image(input_path)
    return np.asarray(
        Image.fromarray(source).resize(
            (
                source.shape[1] * CONFIGURED_OUTPUT_SCALE,
                source.shape[0] * CONFIGURED_OUTPUT_SCALE,
            ),
            resample=Image.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )


class _OnnxRunner:
    def __init__(self, config: Mapping[str, object], base: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime is not installed") from error
        model_path = _resolve_local_path(base, config.get("model_path"), "runtime.model_path")
        if not model_path.is_file():
            raise RuntimeError("Configured ONNX model is missing")
        expected_sha = config.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise RuntimeError("ONNX runtime requires the model's SHA-256")
        if _sha256_path(model_path).lower() != expected_sha.lower():
            raise RuntimeError("Configured ONNX model checksum does not match")
        provider = config.get("provider", "cpu")
        provider_name = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "directml": "DmlExecutionProvider",
        }.get(provider)
        if provider_name is None:
            raise RuntimeError(f"Unsupported ONNX provider: {provider}")
        if provider_name not in ort.get_available_providers():
            raise RuntimeError(f"Configured ONNX provider is unavailable: {provider}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider_name])
        self._input_name = str(config.get("input_name") or self._session.get_inputs()[0].name)
        self._output_name = str(config.get("output_name") or self._session.get_outputs()[0].name)

    def run(self, input_path: Path) -> tuple[np.ndarray, float, int | None]:
        started = time.perf_counter()
        image, peak = _measure_peak_memory(lambda: self._run_materialized(input_path))
        elapsed = (time.perf_counter() - started) * 1000.0
        return image, elapsed, peak

    def _run_materialized(self, input_path: Path) -> np.ndarray:
        source = _load_image(input_path).astype(np.float32) / 255.0
        tensor = np.transpose(source, (2, 0, 1))[None].copy()
        output = self._session.run([self._output_name], {self._input_name: tensor})[0]
        array = np.asarray(output)
        if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] not in (1, 3, 4):
            raise RuntimeError(f"ONNX output has unsupported shape {array.shape}")
        image = np.transpose(array[0], (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        image = np.rint(np.clip(image[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        expected_shape = (source.shape[0] * 2, source.shape[1] * 2)
        if image.shape[:2] != expected_shape:
            raise RuntimeError(f"ONNX output is {image.shape[:2]}, expected {expected_shape}")
        return image


def _validate_local_artifacts(runtime: Mapping[str, object], base: Path) -> None:
    artifacts = runtime.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("Command runtime requires checksummed local artifacts")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise RuntimeError(f"runtime.artifacts[{index}] must be an object")
        path = _resolve_local_path(base, artifact.get("path"), f"runtime.artifacts[{index}].path")
        expected = artifact.get("sha256")
        if not path.is_file() or not isinstance(expected, str) or len(expected) != 64:
            raise RuntimeError("Every command artifact must exist and declare SHA-256")
        if _sha256_path(path).lower() != expected.lower():
            raise RuntimeError(f"Artifact checksum mismatch: {path.name}")


def _command_x2(
    runtime: Mapping[str, object],
    base: Path,
    input_path: Path,
    output_path: Path,
) -> tuple[np.ndarray, float, int | None]:
    if runtime.get("offline_confirmed") is not True:
        raise RuntimeError("Command runtime must explicitly confirm offline operation")
    _validate_local_artifacts(runtime, base)
    argv = runtime.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
        raise RuntimeError("Command runtime argv must be a nonempty string array")
    substitutions = {
        "{input}": str(input_path),
        "{output}": str(output_path),
        "{scale}": str(CONFIGURED_OUTPUT_SCALE),
    }
    command = [substitutions.get(item, item) for item in argv]
    if any("://" in item.lower() for item in command):
        raise RuntimeError("Command runtime arguments must not contain network URLs")
    if "{input}" not in argv or "{output}" not in argv:
        raise RuntimeError("Command runtime argv must contain {input} and {output} placeholders")
    timeout = runtime.get("timeout_seconds", 300)
    if not _finite_nonnegative(timeout) or not 1 <= float(timeout) <= 3600:
        raise RuntimeError("Command timeout_seconds must be in [1, 3600]")
    cwd = base
    if runtime.get("cwd") is not None:
        cwd = _resolve_local_path(base, runtime["cwd"], "runtime.cwd")
        if not cwd.is_dir():
            raise RuntimeError("Command runtime cwd is not a directory")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
    )
    started = time.perf_counter()
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            shell=False,
        )
        peak = 0
        deadline = time.monotonic() + float(timeout)
        while process.poll() is None:
            memory = _rss_bytes(os.getpid(), include_children=True)
            if memory is not None:
                peak = max(peak, memory)
            if time.monotonic() >= deadline:
                process.kill()
                process.wait()
                raise RuntimeError("Command runtime exceeded timeout_seconds")
            time.sleep(MEMORY_SAMPLE_INTERVAL_SECONDS)
        if process.returncode != 0:
            stderr.seek(0)
            detail = stderr.read(2048).decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Command runtime exited {process.returncode}: {detail[:500]}")
    if not output_path.is_file():
        raise RuntimeError("Command runtime did not create its requested output")
    output = _load_image(output_path)
    final_memory = _rss_bytes(os.getpid(), include_children=True)
    if final_memory is not None:
        peak = max(peak, final_memory)
    elapsed = (time.perf_counter() - started) * 1000.0
    return output, elapsed, peak or None


def _evaluate_runner(
    model_id: str,
    runtime_kind: str,
    cases: Sequence[Mapping[str, object]],
    run_case: Any,
    model_version: str | None,
) -> dict[str, object]:
    quality: list[QualityMetrics] = []
    elapsed_values: list[float] = []
    peak_values: list[int] = []
    evidence = hashlib.sha256()
    for case in cases:
        reference = _load_image(case["ground_truth_path"])
        output, elapsed_ms, peak_memory = run_case(case)
        if output.shape[:2] != reference.shape[:2]:
            raise RuntimeError(
                f"Candidate output for {case['case_id']} is {output.shape[:2]}, expected {reference.shape[:2]}"
            )
        observation = evaluate_quality_metrics(reference, output, case["annotations"])
        quality.append(observation)
        elapsed_values.append(float(elapsed_ms))
        if peak_memory is None:
            raise RuntimeError("Peak memory could not be measured on this runtime")
        peak_values.append(int(peak_memory))
        evidence.update(str(case["case_id"]).encode("utf-8"))
        evidence.update(np.ascontiguousarray(output).tobytes())
    metrics: dict[str, float | int | None] = mean_quality_metrics(quality)
    metrics["runtime_ms_mean"] = sum(elapsed_values) / len(elapsed_values)
    metrics["peak_memory_bytes"] = max(peak_values)
    missing = [name for name, value in metrics.items() if value is None]
    if missing:
        return _candidate_row(
            model_id,
            "unmeasured",
            f"Approved downstream adapters are unavailable for: {', '.join(missing)}",
            runtime_kind=runtime_kind,
            model_version=model_version,
            metrics=metrics,
            evidence_sha256=evidence.hexdigest(),
        )
    return _candidate_row(
        model_id,
        "measured",
        None,
        runtime_kind=runtime_kind,
        model_version=model_version,
        metrics=metrics,
        evidence_sha256=evidence.hexdigest(),
    )


def _run_candidate(
    model_id: str,
    config: Mapping[str, object] | None,
    cases: Sequence[Mapping[str, object]],
    manifest_base: Path,
    repository_root: Path,
) -> dict[str, object]:
    version, _ = _manifest_metadata(model_id, repository_root, config)
    if not cases:
        return _candidate_row(
            model_id,
            "unmeasured",
            "No fixed local benchmark cases were configured",
            runtime_kind="unconfigured",
            model_version=version,
        )
    if config is None:
        return _candidate_row(
            model_id,
            "blocked",
            "No local runtime and weight configuration was provided",
            runtime_kind="unconfigured",
            model_version=version,
        )
    if config.get("status") in ("blocked", "unmeasured"):
        reason = config.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise BenchmarkManifestError(f"Candidate {model_id} requires a nonempty reason")
        return _candidate_row(
            model_id,
            str(config["status"]),
            reason.strip(),
            runtime_kind="unconfigured",
            model_version=version,
        )
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        return _candidate_row(
            model_id,
            "blocked",
            "Candidate has no local runtime configuration",
            runtime_kind="unconfigured",
            model_version=version,
        )
    runtime_kind = runtime.get("kind")
    try:
        if runtime_kind == "onnx":
            runner = _OnnxRunner(runtime, manifest_base)
            return _evaluate_runner(
                model_id,
                "onnxruntime",
                cases,
                lambda case: runner.run(case["input_path"]),
                version,
            )
        if runtime_kind == "command":
            with tempfile.TemporaryDirectory(prefix="graphsr-benchmark-") as temporary:
                temporary_root = Path(temporary)

                def run_case(case: Mapping[str, object]) -> tuple[np.ndarray, float, int | None]:
                    output = temporary_root / f"{hashlib.sha256(str(case['case_id']).encode()).hexdigest()}.png"
                    return _command_x2(runtime, manifest_base, case["input_path"], output)

                return _evaluate_runner(model_id, "local_command", cases, run_case, version)
        return _candidate_row(
            model_id,
            "blocked",
            f"Unsupported or missing local runtime kind: {runtime_kind}",
            runtime_kind=str(runtime_kind or "unconfigured"),
            model_version=version,
        )
    except (BenchmarkManifestError, OSError, RuntimeError, ValueError) as error:
        return _candidate_row(
            model_id,
            "blocked",
            str(error) or type(error).__name__,
            runtime_kind=str(runtime_kind or "unconfigured"),
            model_version=version,
        )


def _threshold_value(thresholds: Mapping[str, object], metric: str, direction: str) -> float | None:
    key = f"{metric}_{direction}"
    value = thresholds.get(key)
    if value is None:
        value = thresholds.get(f"{direction}_{metric}")
    if value is None:
        nested = thresholds.get(metric)
        if isinstance(nested, Mapping):
            value = nested.get(direction)
        elif nested is not None:
            value = nested
    if not _finite_nonnegative(value):
        return None
    return float(value)


def select_default(
    candidates: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """Select a default only from complete, downstream-justified evidence."""

    rows = {row.get("model_id"): row for row in candidates}
    if tuple(row.get("model_id") for row in candidates) != REQUIRED_CANDIDATES:
        return {"model_id": None, "status": "none", "reason": "Required candidate order/evidence is incomplete", "gate_checks": {}}
    incomplete = [model_id for model_id in REQUIRED_CANDIDATES if rows[model_id].get("status") != "measured"]
    if incomplete:
        return {
            "model_id": None,
            "status": "none",
            "reason": f"Evidence is unavailable for: {', '.join(incomplete)}",
            "gate_checks": {},
        }
    if not all(_valid_metrics(rows[model_id].get("metrics")) for model_id in REQUIRED_CANDIDATES):
        return {"model_id": None, "status": "none", "reason": "Measured evidence is invalid", "gate_checks": {}}
    required_thresholds = {
        **{name: "min" for name in _HIGHER_IS_BETTER},
        **{name: "max" for name in (*_LOWER_IS_BETTER, *_RUNTIME_METRICS)},
    }
    normalized = {
        name: _threshold_value(thresholds, name, direction)
        for name, direction in required_thresholds.items()
    }
    missing_thresholds = [name for name, value in normalized.items() if value is None]
    if missing_thresholds:
        return {
            "model_id": None,
            "status": "none",
            "reason": f"Downstream thresholds are missing or invalid for: {', '.join(missing_thresholds)}",
            "gate_checks": {},
        }
    baseline = rows[_BASELINE_ID]["metrics"]
    gate_checks: dict[str, dict[str, bool]] = {}
    eligible: list[tuple[float, str]] = []
    for model_id in REQUIRED_CANDIDATES[:-1]:
        metrics = rows[model_id]["metrics"]
        checks: dict[str, bool] = {}
        for name in _HIGHER_IS_BETTER:
            checks[f"{name}_absolute"] = float(metrics[name]) >= float(normalized[name])
            checks[f"{name}_no_regression"] = float(metrics[name]) >= float(baseline[name])
        for name in _LOWER_IS_BETTER:
            checks[f"{name}_absolute"] = float(metrics[name]) <= float(normalized[name])
            checks[f"{name}_no_regression"] = float(metrics[name]) <= float(baseline[name])
        for name in _RUNTIME_METRICS:
            checks[f"{name}_absolute"] = float(metrics[name]) <= float(normalized[name])
        checks["numeric_ocr_strict_improvement"] = (
            float(metrics["numeric_ocr_exact_match"])
            > float(baseline["numeric_ocr_exact_match"])
        )
        checks["marker_f1_strict_improvement"] = (
            float(metrics["marker_center_f1"]) > float(baseline["marker_center_f1"])
        )
        gate_checks[model_id] = checks
        if all(checks.values()):
            score = sum(float(metrics[name]) - float(baseline[name]) for name in _HIGHER_IS_BETTER)
            score += sum(float(baseline[name]) - float(metrics[name]) for name in _LOWER_IS_BETTER)
            eligible.append((score, model_id))
    if not eligible:
        return {
            "model_id": None,
            "status": "none",
            "reason": "No candidate passed all downstream gates and baseline comparisons",
            "gate_checks": gate_checks,
        }
    eligible.sort(key=lambda item: (-item[0], REQUIRED_CANDIDATES.index(item[1])))
    selected = eligible[0][1]
    return {
        "model_id": selected,
        "status": "selected",
        "reason": "All evidence, absolute gates, no-regression checks, and required downstream improvements passed",
        "gate_checks": gate_checks,
    }


def run_benchmark(manifest: Path | str) -> dict[str, object]:
    """Execute one fixed local benchmark plan and return its report."""

    manifest_path = Path(manifest).resolve()
    source = _load_json(manifest_path)
    plan = _normalize_benchmark_manifest(source, manifest_path)
    repository_root = Path(__file__).resolve().parents[2]
    cases = plan["cases"]
    candidates_config = plan["candidates"]
    rows: list[dict[str, object]] = []
    for model_id in REQUIRED_CANDIDATES[:-1]:
        rows.append(
            _run_candidate(
                model_id,
                candidates_config.get(model_id),
                cases,
                manifest_path.parent,
                repository_root,
            )
        )
    if not cases:
        baseline = _candidate_row(
            _BASELINE_ID,
            "unmeasured",
            "No fixed local benchmark cases were configured",
            runtime_kind="pillow_bicubic",
            model_version="Pillow",
        )
    else:
        try:
            baseline = _evaluate_runner(
                _BASELINE_ID,
                "pillow_bicubic",
                cases,
                lambda case: _bicubic_x2(case["input_path"]),
                Image.__version__ if hasattr(Image, "__version__") else None,
            )
        except (OSError, RuntimeError, ValueError) as error:
            baseline = _candidate_row(
                _BASELINE_ID,
                "blocked",
                str(error) or type(error).__name__,
                runtime_kind="pillow_bicubic",
            )
    rows.append(baseline)
    selection = select_default(rows, plan["thresholds"])
    dataset_evidence = [
        {
            "case_id": case["case_id"],
            "input_sha256": _sha256_path(case["input_path"]),
            "ground_truth_sha256": _sha256_path(case["ground_truth_path"]),
            "annotations_sha256": _canonical_sha256(case["annotations"]),
        }
        for case in cases
    ]
    status = "measured" if all(row["status"] == "measured" for row in rows) else "incomplete"
    # A complete, truthful no-default report is a successful benchmark
    # invocation.  Selection is scientific evidence state, not process health.
    exit_code = 0
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": status,
        "source_kind": plan["source_kind"],
        "source_manifest_sha256": _sha256_path(manifest_path),
        "dataset": {
            "dataset_id": plan["dataset_id"],
            "status": "measured" if cases else "unmeasured",
            "case_count": len(cases),
            "evidence_sha256": _canonical_sha256(dataset_evidence) if cases else None,
        },
        "metric_names": list(METRIC_NAMES),
        "measurement_contract": dict(MEASUREMENT_CONTRACT),
        "candidates": rows,
        "selection": selection,
        "network_access": "disabled_by_contract",
        "exit_code": exit_code,
    }


def _invalid_report(error: Exception) -> dict[str, object]:
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "status": "invalid",
        "error": {
            "code": "INVALID_LOCAL_MANIFEST",
            "message": str(error) or type(error).__name__,
        },
        "exit_code": 2,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Local benchmark or GraphSR model manifest")
    parser.add_argument("--output", type=Path, help="Optional path for the stable JSON report")
    arguments = parser.parse_args(argv)
    try:
        report = run_benchmark(arguments.manifest)
    except (BenchmarkManifestError, OSError, RuntimeError, ValueError) as error:
        report = _invalid_report(error)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if arguments.output is not None:
        try:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(payload, encoding="utf-8")
        except OSError as error:
            report = _invalid_report(BenchmarkManifestError(f"Could not write output: {error}"))
            payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    sys.stdout.write(payload)
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "BenchmarkManifestError",
    "METRIC_NAMES",
    "REQUIRED_CANDIDATES",
    "main",
    "run_benchmark",
    "select_default",
]
