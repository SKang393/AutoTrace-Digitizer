# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang

"""Freeze and execute the one-run bounded-probability PP-OCRv5 gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

from ml.markers.gate_seal import (
    GateSeal,
    acquire_gate_seal,
    complete_gate_seal,
    sha256_file,
)
from ml.ocr.official_bakeoff import structure_consensus_evaluate as base


PROFILE = "graphreader-ocr-structure-consensus-bounded-public-gate-v2"
SPLIT_SCHEMA = "graphreader.ocr-structure-consensus-bounded-split.v2"
CORE_SCHEMA = "graphreader.ocr-structure-consensus-bounded-core-predictions.v2"
PREDICTIONS_SCHEMA = "graphreader.ocr-structure-consensus-bounded-predictions.v2"
RUNTIME_SCHEMA = "graphreader.ocr-structure-consensus-bounded-runtime-results.v2"
REPORT_SCHEMA = "graphreader.ocr-structure-consensus-bounded-production-gate.v2"
PROTOCOL_SCHEMA = "graphreader.ocr-structure-consensus-bounded-gate-protocol.v2"
CONFIG_SCHEMA = "graphreader.ocr-structure-consensus-bounded-evaluation-config.v1"
COMPOSITION_ID = "graph-structure-consensus-bounded-v2"
OUTPUT_ACTIVATION = "probability_with_1e-5_clamp"
PROBABILITY_TOLERANCE = 1e-5
RENDER_INDEX_OFFSET = 100_003
GATE_TASK = "ocr-detection"
GATE_REVISION = "official-structure-consensus-bounded-v2"
EXPECTED_PRIOR_SPLITS = 2
EVALUATOR_SOURCE_PATHS = (
    Path("ml/ocr/official_bakeoff/structure_consensus_v2_evaluate.py"),
    Path("ml/ocr/official_bakeoff/structure_consensus_evaluate.py"),
    Path("ml/ocr/production_gate.py"),
    Path("ml/markers/gate_seal.py"),
)


ProductionGateError = base.ProductionGateError
load_strict_json = base.load_strict_json
canonical_json_bytes = base.canonical_json_bytes
hash_bytes = base.hash_bytes
hash_file = base.hash_file


_detector_observations: list[dict[str, float | int]] = []


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionGateError(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_protocol() -> Path:
    return Path(__file__).resolve().with_name("STRUCTURE_CONSENSUS_V2_GATE_PROTOCOL.json")


def _default_evaluation_config() -> Path:
    return Path(__file__).resolve().with_name("STRUCTURE_CONSENSUS_V2_EVALUATION_CONFIG.json")


def _default_metrics_evaluator() -> Path:
    return Path(__file__).resolve().parents[1] / "production_gate.py"


def _sha256_hex(value: object, label: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text.lower()),
        f"{label} must be SHA-256.",
    )
    return text.lower()


def _activation_gate_config() -> dict[str, object]:
    return {
        "profile": PROFILE,
        "composition_id": COMPOSITION_ID,
        "execution_provider": "CPUExecutionProvider",
        "output_activation": OUTPUT_ACTIVATION,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "detector_postprocess": "db_postprocess_v1",
        "marker_stage_evidence_required": True,
        "scope": "public_synthetic",
        "private_data": False,
        "chandler_used": False,
    }


def validate_protocol(
    protocol_path: Path,
    metrics_evaluator_path: Path,
    workflow_path: Path | None = None,
) -> dict[str, Any]:
    protocol = load_strict_json(protocol_path)
    workflow_path = workflow_path or Path(__file__).resolve()
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "Bounded gate protocol schema is invalid.")
    _require(protocol.get("profile") == PROFILE, "Bounded gate profile changed.")
    _require(
        protocol.get("status") == "frozen_before_fixture_generation_and_inference",
        "Bounded gate protocol is not frozen before inference.",
    )
    _require(protocol.get("scope") == "public_synthetic", "OCR scope must be public synthetic.")
    _require(protocol.get("private_data") is False, "Private data is prohibited.")
    _require(protocol.get("chandler_used") is False, "Chandler is prohibited before private validation.")
    _require(
        protocol.get("selection_locked_before_inference") is True,
        "Candidate selection must be locked before inference.",
    )
    _require(
        protocol.get("metrics_evaluator_sha256") == hash_file(metrics_evaluator_path),
        "Frozen metrics evaluator changed.",
    )
    _require(
        protocol.get("execution_workflow_sha256") == hash_file(workflow_path),
        "Frozen bounded workflow changed.",
    )
    sources = protocol.get("reviewed_source_sha256")
    _require(isinstance(sources, dict) and sources, "Reviewed source inventory is missing.")
    root = _repo_root()
    for relative, expected in sources.items():
        _require(isinstance(relative, str), "Reviewed source path is invalid.")
        source = (root / relative).resolve()
        _require(source.is_relative_to(root) and source.is_file(), f"Reviewed source is missing: {relative}")
        _require(hash_file(source) == expected, f"Reviewed source changed: {relative}")
    prior = protocol.get("prior_exposed_splits_forbidden")
    _require(
        isinstance(prior, list) and len(prior) == EXPECTED_PRIOR_SPLITS,
        "Both prior exposed split denials are required.",
    )
    labels: set[str] = set()
    for entry in prior:
        _require(isinstance(entry, dict), "Prior exposed split denial is invalid.")
        label = str(entry.get("label"))
        _require(label and label not in labels, "Prior exposed split label is invalid.")
        labels.add(label)
        _sha256_hex(entry.get("split_sha256"), f"{label} split")
        _sha256_hex(entry.get("fixture_archive_sha256"), f"{label} fixture archive")
        _sha256_hex(entry.get("source_inventory_sha256"), f"{label} source inventory")
    candidate = protocol.get("candidate")
    _require(isinstance(candidate, dict), "Bounded candidate is missing.")
    _require(candidate.get("composition_id") == COMPOSITION_ID, "Composition identity changed.")
    _require(candidate.get("output_activation") == OUTPUT_ACTIVATION, "Output activation changed.")
    _require(
        float(candidate.get("probability_tolerance", -1)) == PROBABILITY_TOLERANCE,
        "Probability tolerance changed.",
    )
    new_split = protocol.get("new_split")
    _require(isinstance(new_split, dict), "New split contract is missing.")
    _require(
        int(new_split.get("render_index_offset", -1)) == RENDER_INDEX_OFFSET,
        "Render index offset changed.",
    )
    authorization = protocol.get("evaluation_authorization")
    _require(isinstance(authorization, dict), "Evaluation authorization boundary is missing.")
    _require(authorization.get("required_config_path") == _default_evaluation_config().name, "Authorization path changed.")
    _require(authorization.get("canonical_gate_task") == GATE_TASK, "Canonical gate task changed.")
    _require(authorization.get("revision") == GATE_REVISION, "Gate revision changed.")
    _require(
        authorization.get("candidate_hash_key_schema")
        == [
            "detector_onnx_sha256",
            "recognizer_onnx_sha256",
            "protocol_sha256",
            "sealed_split_sha256",
        ],
        "Candidate hash key schema changed.",
    )
    _require(
        protocol.get("experiment_budget")
        == {
            "fixture_freezes": 1,
            "official_composition_evaluations": 1,
            "split_regeneration_after_inference": 0,
            "threshold_changes_after_inference": 0,
            "workflow_changes_after_inference": 0,
        },
        "Bounded composition experiment budget changed.",
    )
    return protocol


def _source_inventory_sha256(cases: Sequence[dict[str, Any]]) -> str:
    values = sorted(_sha256_hex(case.get("source_sha256"), "Prior source") for case in cases)
    return hash_bytes("\n".join(values).encode("utf-8"))


def _prior_exposed_identity(
    protocol: dict[str, Any],
    prior_frozen_roots: Sequence[Path],
) -> tuple[set[str], set[str]]:
    entries = protocol["prior_exposed_splits_forbidden"]
    _require(len(prior_frozen_roots) == len(entries), "Every prior exposed split root is required.")
    source_hashes: set[str] = set()
    case_ids: set[str] = set()
    for entry, frozen_root in zip(entries, prior_frozen_roots, strict=True):
        split_path = frozen_root / "split.json"
        archive_path = frozen_root / "fixtures.zip"
        _require(split_path.is_file() and archive_path.is_file(), f"Prior frozen root is incomplete: {frozen_root}")
        _require(hash_file(split_path) == entry["split_sha256"], f"Prior split changed: {entry['label']}")
        _require(
            hash_file(archive_path) == entry["fixture_archive_sha256"],
            f"Prior fixture archive changed: {entry['label']}",
        )
        split = load_strict_json(split_path)
        cases = split.get("cases")
        _require(isinstance(cases, list) and cases, f"Prior split cases are missing: {entry['label']}")
        _require(
            _source_inventory_sha256(cases) == entry["source_inventory_sha256"],
            f"Prior source inventory changed: {entry['label']}",
        )
        source_hashes.update(str(case["source_sha256"]) for case in cases)
        case_ids.update(str(case["case_id"]) for case in cases)
    return source_hashes, case_ids


def freeze_split(
    output_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
    font_path: Path,
    prior_frozen_roots: Sequence[Path],
) -> dict[str, Any]:
    protocol = validate_protocol(protocol_path, metrics_evaluator_path)
    prior_source_hashes, prior_case_ids = _prior_exposed_identity(protocol, prior_frozen_roots)
    _require(font_path.is_file(), f"Frozen renderer font is missing: {font_path}")
    _require(not output_root.exists(), "Frozen output root already exists.")
    cases: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(("validation", "sealed_test")):
        for index in range(base.EXPECTED_TEXT_PER_PARTITION):
            family = base.TEXT_FAMILIES[index % len(base.TEXT_FAMILIES)]
            role = base.TEXT_ROLES[(index // len(base.TEXT_FAMILIES)) % len(base.TEXT_ROLES)]
            context = base.GRAPH_CONTEXT_FAMILIES[(index * 5 + partition_index) % len(base.GRAPH_CONTEXT_FAMILIES)]
            degradation = base.DEGRADATIONS[(index * 3 + partition_index) % len(base.DEGRADATIONS)]
            case_id = f"{partition}-v2-text-{index:03d}"
            render_index = RENDER_INDEX_OFFSET + index + (partition_index * base.EXPECTED_TEXT_PER_PARTITION)
            image, display, truth, truth_box, rectangles = base._render_case(
                case_id,
                "text",
                family,
                role,
                context,
                degradation,
                render_index,
                font_path,
            )
            cases.append(_write_case(output_root, case_id, partition, "text", family, role, context,
                                     degradation, display, truth, truth_box, rectangles, image))
        for index in range(base.EXPECTED_EXCLUSIONS_PER_PARTITION):
            context = base.GRAPH_CONTEXT_FAMILIES[(index * 7 + partition_index) % len(base.GRAPH_CONTEXT_FAMILIES)]
            degradation = base.DEGRADATIONS[(index + partition_index) % len(base.DEGRADATIONS)]
            case_id = f"{partition}-v2-exclusion-{index:03d}"
            render_index = (
                RENDER_INDEX_OFFSET
                + 1_000
                + index
                + (partition_index * base.EXPECTED_EXCLUSIONS_PER_PARTITION)
            )
            image, _, _, _, rectangles = base._render_case(
                case_id,
                "exclusion",
                "exclusion",
                "other",
                context,
                degradation,
                render_index,
                font_path,
            )
            cases.append(_write_case(output_root, case_id, partition, "exclusion", "exclusion", "other", context,
                                     degradation, "", "", None, rectangles, image))
    _require(len(cases) == base.EXPECTED_CASES, "Frozen split case count is invalid.")
    new_source_hashes = {str(case["source_sha256"]) for case in cases}
    new_case_ids = {str(case["case_id"]) for case in cases}
    _require(new_source_hashes.isdisjoint(prior_source_hashes), "New fixture bytes reused a prior exposed source.")
    _require(new_case_ids.isdisjoint(prior_case_ids), "New case IDs reused a prior exposed split.")
    fixture_archive = base.locked.build_fixture_archive(output_root, cases)
    prior_bindings = [
        {
            "label": entry["label"],
            "split_sha256": entry["split_sha256"],
            "fixture_archive_sha256": entry["fixture_archive_sha256"],
            "source_inventory_sha256": entry["source_inventory_sha256"],
        }
        for entry in protocol["prior_exposed_splits_forbidden"]
    ]
    split = {
        "schema": SPLIT_SCHEMA,
        "profile": PROFILE,
        "scope": "public_synthetic",
        "sealed": True,
        "selection_locked_before_inference": True,
        "private_data": False,
        "chandler_used": False,
        "protocol_sha256": hash_file(protocol_path),
        "evaluator_source_sha256": hash_file(metrics_evaluator_path),
        "workflow_source_sha256": hash_file(Path(__file__)),
        "fixture_archive_sha256": hash_bytes(fixture_archive),
        "prior_exposed_splits": prior_bindings,
        "source_hash_disjoint_from_prior_exposed_splits": True,
        "case_id_disjoint_from_prior_exposed_splits": True,
        "renderer": {
            "family": "graphreader-structure-context-renderer-v2",
            "seed": protocol["new_split"]["renderer_seed"],
            "render_index_offset": RENDER_INDEX_OFFSET,
            "font_sha256": hash_file(font_path),
            "width": 320,
            "height": 160,
        },
        "cases": cases,
    }
    split_bytes = canonical_json_bytes(split)
    for forbidden in protocol["prior_exposed_splits_forbidden"]:
        _require(hash_bytes(split_bytes) != forbidden["split_sha256"], "New split reused an exposed split.")
        _require(
            hash_bytes(fixture_archive) != forbidden["fixture_archive_sha256"],
            "New fixtures reused an exposed archive.",
        )
    base._write_new(output_root / "fixtures.zip", fixture_archive)
    base._write_new(output_root / "split.json", split_bytes)
    return {
        "case_count": len(cases),
        "sealed_split_sha256": hash_bytes(split_bytes),
        "fixture_archive_sha256": hash_bytes(fixture_archive),
        "source_inventory_sha256": _source_inventory_sha256(cases),
        "prior_source_overlap_count": 0,
        "prior_case_id_overlap_count": 0,
        "split": split,
    }


def _write_case(
    output_root: Path,
    case_id: str,
    partition: str,
    kind: str,
    family: str,
    role: str,
    context: str,
    degradation: str,
    display: str,
    truth: str,
    truth_box: Any | None,
    rectangles: list[dict[str, Any]],
    image: Any,
) -> dict[str, Any]:
    path = output_root / "assets" / f"{case_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)
    with __import__("PIL.Image", fromlist=["Image"]).open(path) as loaded:
        exact = loaded.convert("RGB")
    source_bgr = base._source_bgr(exact)
    detector_bgr = base._masked_bgr(exact, rectangles)
    return {
        "case_id": case_id,
        "partition": partition,
        "kind": kind,
        "family": family,
        "graph_context_family": context,
        "degradation_family": degradation,
        "display_text": display,
        "truth_text": truth,
        "truth_role": role,
        "truth_bbox": truth_box.to_json() if truth_box is not None else None,
        "expected_region_count": 1 if kind == "text" else 0,
        "source_path": f"assets/{case_id}.png",
        "source_sha256": hash_file(path),
        "source_width": exact.width,
        "source_height": exact.height,
        "source_bgr_sha256": hash_bytes(source_bgr),
        "detector_image_bgr_sha256": hash_bytes(detector_bgr),
        "mask_rectangles": rectangles,
    }


@contextmanager
def _configured_base() -> Iterator[None]:
    replacements: dict[str, object] = {
        "PROFILE": PROFILE,
        "SPLIT_SCHEMA": SPLIT_SCHEMA,
        "CORE_SCHEMA": CORE_SCHEMA,
        "PREDICTIONS_SCHEMA": PREDICTIONS_SCHEMA,
        "RUNTIME_SCHEMA": RUNTIME_SCHEMA,
        "REPORT_SCHEMA": REPORT_SCHEMA,
        "COMPOSITION_ID": COMPOSITION_ID,
        "validate_protocol": validate_protocol,
        "detect_regions": detect_regions,
        "__file__": __file__,
    }
    original = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(base, name, value)


def verify_frozen_split(
    frozen_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
    prior_frozen_roots: Sequence[Path] = (),
) -> dict[str, Any]:
    with _configured_base():
        verification = base.verify_frozen_split(frozen_root, protocol_path, metrics_evaluator_path)
    protocol = verification["protocol"]
    split = verification["split"]
    expected_prior = protocol["prior_exposed_splits_forbidden"]
    _require(split.get("prior_exposed_splits") == expected_prior, "Prior split bindings changed.")
    _require(
        split.get("source_hash_disjoint_from_prior_exposed_splits") is True,
        "Source disjointness evidence is missing.",
    )
    _require(
        split.get("case_id_disjoint_from_prior_exposed_splits") is True,
        "Case-ID disjointness evidence is missing.",
    )
    if prior_frozen_roots:
        prior_source_hashes, prior_case_ids = _prior_exposed_identity(protocol, prior_frozen_roots)
        cases = split["cases"]
        _require(
            {str(case["source_sha256"]) for case in cases}.isdisjoint(prior_source_hashes),
            "Frozen source bytes overlap prior exposed fixtures.",
        )
        _require(
            {str(case["case_id"]) for case in cases}.isdisjoint(prior_case_ids),
            "Frozen case IDs overlap prior exposed fixtures.",
        )
    return verification


def detect_regions(session: Any, masked_bgr: bytes, width: int, height: int) -> base.DetectionEvidence:
    import numpy as np

    tensor = base.detector_tensor(masked_bgr, width, height)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    _require(len(inputs) == 1 and len(outputs) == 1, "Detector ONNX contract changed.")
    started = perf_counter()
    output = np.asarray(session.run([outputs[0].name], {inputs[0].name: tensor})[0], dtype=np.float32)
    duration_ms = (perf_counter() - started) * 1000.0
    _require(output.shape == (1, 1, tensor.shape[2], tensor.shape[3]), "Detector output shape changed.")
    _require(np.isfinite(output).all(), "Detector output contains non-finite values.")
    observed_minimum = float(output.min())
    observed_maximum = float(output.max())
    lower_drift = max(0.0, -observed_minimum)
    upper_drift = max(0.0, observed_maximum - 1.0)
    maximum_drift = max(lower_drift, upper_drift)
    _require(
        maximum_drift <= PROBABILITY_TOLERANCE,
        "Detector output exceeded the fixed 1e-5 probability tolerance: "
        f"minimum={observed_minimum:.9g}; maximum={observed_maximum:.9g}.",
    )
    clamped_value_count = int(np.count_nonzero((output < 0.0) | (output > 1.0)))
    _detector_observations.append(
        {
            "observed_minimum": observed_minimum,
            "observed_maximum": observed_maximum,
            "maximum_boundary_drift": maximum_drift,
            "clamped_value_count": clamped_value_count,
        }
    )
    probabilities = np.clip(output, np.float32(0.0), np.float32(1.0))
    model_regions = base.db_model_regions(probabilities, width, height)
    candidates = base.connected_component_candidates(masked_bgr, width, height)
    matches, final_regions = base.compose_consensus(model_regions, candidates)
    return base.DetectionEvidence(
        model_regions,
        candidates,
        matches,
        final_regions,
        hash_bytes(tensor.tobytes(order="C")),
        tuple(int(value) for value in tensor.shape),
        hash_bytes(output.tobytes(order="C")),
        tuple(int(value) for value in output.shape),
        duration_ms,
    )


def _activation_summary() -> dict[str, object]:
    _require(_detector_observations, "No detector output observations were recorded.")
    return {
        "output_activation": OUTPUT_ACTIVATION,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "detector_call_count": len(_detector_observations),
        "observed_minimum": min(float(item["observed_minimum"]) for item in _detector_observations),
        "observed_maximum": max(float(item["observed_maximum"]) for item in _detector_observations),
        "maximum_boundary_drift": max(
            float(item["maximum_boundary_drift"]) for item in _detector_observations
        ),
        "clamped_value_count": sum(int(item["clamped_value_count"]) for item in _detector_observations),
        "material_drift_rejected": True,
        "non_finite_rejected": True,
    }


def _bind_activation_evidence(output_root: Path, seal: GateSeal) -> dict[str, Any]:
    activation = _activation_summary()
    opened_sha256 = sha256_file(seal.opened_path)
    seal_evidence = {
        "key": seal.key,
        "opened_sha256": opened_sha256,
        "binding": seal.binding,
    }
    core_path = output_root / "core-predictions.json"
    predictions_path = output_root / "predictions.json"
    runtime_path = output_root / "runtime-results.json"
    report_path = output_root / "report.json"
    core = load_strict_json(core_path)
    core["detector_output_activation"] = activation
    core["public_gate_seal"] = seal_evidence
    core_bytes = canonical_json_bytes(core)
    core_path.write_bytes(core_bytes)
    core_sha256 = hash_bytes(core_bytes)
    predictions = load_strict_json(predictions_path)
    predictions["core_predictions_sha256"] = core_sha256
    predictions["detector_output_activation"] = activation
    predictions["public_gate_seal"] = seal_evidence
    predictions_bytes = canonical_json_bytes(predictions)
    predictions_path.write_bytes(predictions_bytes)
    predictions_sha256 = hash_bytes(predictions_bytes)
    runtime = load_strict_json(runtime_path)
    runtime["core_predictions_sha256"] = core_sha256
    runtime["predictions_sha256"] = predictions_sha256
    runtime["detector_output_activation"] = activation
    runtime["public_gate_seal"] = seal_evidence
    provenance = runtime.get("execution_provenance")
    _require(isinstance(provenance, dict), "Runtime execution provenance is missing.")
    provenance["detector_output_activation"] = activation
    runtime_bytes = canonical_json_bytes(runtime)
    runtime_path.write_bytes(runtime_bytes)
    runtime_sha256 = hash_bytes(runtime_bytes)
    report = load_strict_json(report_path)
    report["core_predictions_sha256"] = core_sha256
    report["predictions_sha256"] = predictions_sha256
    report["runtime_results_sha256"] = runtime_sha256
    report["detector_output_activation"] = activation
    report["public_gate_seal"] = seal_evidence
    resources = report.get("reviewed_resources")
    _require(isinstance(resources, dict), "Reviewed report resources are missing.")
    resources["core_predictions"] = base._embedded_resource(
        "application/json", core_bytes, "Core predictions"
    )
    resources["predictions"] = base._embedded_resource(
        "application/json", predictions_bytes, "Predictions"
    )
    resources["runtime_results"] = base._embedded_resource(
        "application/json", runtime_bytes, "Runtime results"
    )
    report_path.write_bytes(canonical_json_bytes(report))
    return report


def _read_evaluation_config(config_path: Path, protocol_path: Path) -> dict[str, Any]:
    config = load_strict_json(config_path)
    _require(config.get("schema") == CONFIG_SCHEMA, "Evaluation config schema is invalid.")
    _require(config.get("status") == "authorized_after_single_freeze", "Evaluation is not authorized.")
    _require(config.get("task") == GATE_TASK, "Evaluation task changed.")
    _require(config.get("revision") == GATE_REVISION, "Evaluation revision changed.")
    _require(config.get("private_data") is False, "Evaluation config permits private data.")
    _require(config.get("chandler_used") is False, "Evaluation config permits Chandler.")
    _require(config.get("evaluation_count") == 1, "Evaluation count must be exactly one.")
    _require(config.get("expected_dataset_manifest_sha256") == config.get("sealed_split_sha256"),
             "Dataset manifest binding changed.")
    _require(config.get("protocol_sha256") == hash_file(protocol_path), "Authorized protocol changed.")
    _require(config.get("evaluator_source_paths") == [path.as_posix() for path in EVALUATOR_SOURCE_PATHS],
             "Evaluator source path inventory changed.")
    _require(config.get("gate_config") == _activation_gate_config(), "Authorized gate configuration changed.")
    candidate_hashes = config.get("candidate_hashes")
    _require(isinstance(candidate_hashes, dict), "Authorized candidate hashes are missing.")
    _require(
        list(candidate_hashes) == [
            "detector_onnx_sha256",
            "recognizer_onnx_sha256",
            "protocol_sha256",
            "sealed_split_sha256",
        ],
        "Authorized candidate hash key order changed.",
    )
    for label, value in candidate_hashes.items():
        _sha256_hex(value, label)
    _require(candidate_hashes["protocol_sha256"] == config["protocol_sha256"], "Candidate protocol hash changed.")
    _require(candidate_hashes["sealed_split_sha256"] == config["sealed_split_sha256"],
             "Candidate split hash changed.")
    return config


def evaluate_once(
    frozen_root: Path,
    protocol_path: Path,
    metrics_evaluator_path: Path,
    conversion_report_path: Path,
    source_root: Path,
    output_root: Path,
    evaluation_config_path: Path,
    prior_frozen_roots: Sequence[Path],
    failure_record_path: Path,
    *,
    session_factory: Callable[[Path], Any] = base.locked._cpu_session,
    parity_runner: Callable[[dict[str, Any], Path], list[dict[str, Any]]] = base.locked._direct_parity_pairs,
) -> tuple[dict[str, Any], GateSeal, Path]:
    protocol = validate_protocol(protocol_path, metrics_evaluator_path)
    config = _read_evaluation_config(evaluation_config_path, protocol_path)
    _require(not output_root.exists(), "One-run OCR evaluation output already exists.")
    _require(not failure_record_path.exists(), "One-run OCR failure record already exists.")
    _require(conversion_report_path.is_file(), "Conversion report is missing.")
    _require(source_root.is_dir(), "Extracted official source root is missing.")
    _require(len(prior_frozen_roots) == EXPECTED_PRIOR_SPLITS, "Both prior frozen roots are required.")
    candidate_hashes = dict(config["candidate_hashes"])
    _require(candidate_hashes["detector_onnx_sha256"] == protocol["candidate"]["detector_onnx_sha256"],
             "Authorized detector changed.")
    _require(candidate_hashes["recognizer_onnx_sha256"] == protocol["candidate"]["recognizer_onnx_sha256"],
             "Authorized recognizer changed.")
    seal = acquire_gate_seal(
        repo_root=_repo_root(),
        task=GATE_TASK,
        revision=GATE_REVISION,
        candidate_hashes=candidate_hashes,
        dataset_manifest_sha256=str(config["expected_dataset_manifest_sha256"]),
        split_config_path=evaluation_config_path.resolve().relative_to(_repo_root()),
        evaluator_source_paths=EVALUATOR_SOURCE_PATHS,
        gate_config=_activation_gate_config(),
    )
    _detector_observations.clear()
    try:
        _require(
            hash_file(frozen_root / "split.json") == config["sealed_split_sha256"],
            "Authorized split changed.",
        )
        _require(
            hash_file(frozen_root / "fixtures.zip") == config["fixture_archive_sha256"],
            "Authorized fixture archive changed.",
        )
        verify_frozen_split(frozen_root, protocol_path, metrics_evaluator_path, prior_frozen_roots)
        with _configured_base():
            base.evaluate_official_candidate(
                frozen_root,
                protocol_path,
                metrics_evaluator_path,
                conversion_report_path,
                source_root,
                output_root,
                session_factory=session_factory,
                parity_runner=parity_runner,
            )
        report = _bind_activation_evidence(output_root, seal)
        result_path = complete_gate_seal(
            seal,
            status=str(report["status"]),
            report_sha256=hash_file(output_root / "report.json"),
        )
        return report, seal, result_path
    except Exception as error:
        failure = {
            "schema": "graphreader.ocr-structure-consensus-bounded-failure.v1",
            "status": "failed_runner",
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "private_data": False,
            "chandler_used": False,
            "protocol_sha256": hash_file(protocol_path),
            "sealed_split_sha256": str(config["sealed_split_sha256"]),
            "fixture_archive_sha256": str(config["fixture_archive_sha256"]),
            "output_activation": OUTPUT_ACTIVATION,
            "probability_tolerance": PROBABILITY_TOLERANCE,
            "gate_seal_key": seal.key,
            "opened_seal_sha256": sha256_file(seal.opened_path),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        base._write_new(failure_record_path, canonical_json_bytes(failure))
        complete_gate_seal(
            seal,
            status="failed_runner",
            report_sha256=hash_file(failure_record_path),
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output-root", required=True, type=Path)
    freeze.add_argument("--prior-frozen-root", required=True, action="append", type=Path)
    freeze.add_argument("--protocol", type=Path, default=_default_protocol())
    freeze.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    freeze.add_argument(
        "--font",
        type=Path,
        default=_repo_root() / "src" / "GraphReader.App" / "Assets" / "Fonts" / "NotoSans-Regular.ttf",
    )
    verify = subparsers.add_parser("verify-freeze")
    verify.add_argument("--frozen-root", required=True, type=Path)
    verify.add_argument("--prior-frozen-root", required=True, action="append", type=Path)
    verify.add_argument("--protocol", type=Path, default=_default_protocol())
    verify.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--frozen-root", required=True, type=Path)
    evaluate.add_argument("--prior-frozen-root", required=True, action="append", type=Path)
    evaluate.add_argument("--protocol", type=Path, default=_default_protocol())
    evaluate.add_argument("--metrics-evaluator", type=Path, default=_default_metrics_evaluator())
    evaluate.add_argument("--conversion-report", required=True, type=Path)
    evaluate.add_argument("--source-root", required=True, type=Path)
    evaluate.add_argument("--output-root", required=True, type=Path)
    evaluate.add_argument("--evaluation-config", type=Path, default=_default_evaluation_config())
    evaluate.add_argument("--failure-record", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_split(
                args.output_root,
                args.protocol,
                args.metrics_evaluator,
                args.font,
                args.prior_frozen_root,
            )
            print(
                canonical_json_bytes({key: value for key, value in result.items() if key != "split"})
                .decode("utf-8")
                .strip()
            )
        elif args.command == "verify-freeze":
            result = verify_frozen_split(
                args.frozen_root,
                args.protocol,
                args.metrics_evaluator,
                args.prior_frozen_root,
            )
            print(canonical_json_bytes({"status": "pass", "case_count": len(result["split"]["cases"])})
                  .decode("utf-8").strip())
        else:
            report, seal, result_path = evaluate_once(
                args.frozen_root,
                args.protocol,
                args.metrics_evaluator,
                args.conversion_report,
                args.source_root,
                args.output_root,
                args.evaluation_config,
                args.prior_frozen_root,
                args.failure_record,
            )
            print(
                canonical_json_bytes(
                    {
                        "status": report["status"],
                        "production_approval": report["production_approval"],
                        "gate_seal_key": seal.key,
                        "result_seal_sha256": hash_file(result_path),
                    }
                ).decode("utf-8").strip()
            )
            return 0 if report["production_approval"] else 2
    except (ProductionGateError, OSError, RuntimeError, ValueError) as error:
        print(f"BLOCKED: {error}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
