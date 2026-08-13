# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ml.markers.gate_seal import canonical_json_bytes, sha256_bytes, sha256_file, source_bundle_sha256
from .dataset import build_split, proposal_summary, save_sealed_archive, split_fingerprint
from .protocol import PUBLIC_REVISION, REVISION, TASK, VALIDATION_REVISION, protocol_configuration
from .sealed_gate import EVALUATOR_SOURCE_PATHS, GATE_CONFIG
from .validation_gate import EVALUATOR_SOURCE_PATHS as VALIDATION_SOURCES, GATE_CONFIG as VALIDATION_GATE_CONFIG


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path("ml/ocr/production_composition_v5")
PRIVATE_ROOT = ROOT / "artifacts/composition-v5-freeze"
SPLIT_SOURCES = (
    ROOT / "dataset.py", ROOT / "prepare_split.py", ROOT / "protocol.py",
    Path("ml/ocr/production_composition_v1/dataset.py"),
    Path("ml/ocr/component_context_detector_v7/dataset.py"),
    Path("ml/ocr/component_region_detector_v6/dataset.py"),
)


def freeze_split() -> dict[str, str]:
    private = REPO_ROOT / PRIVATE_ROOT
    targets = [
        ROOT / "PROTOCOL.json", ROOT / "VALIDATION_SEAL.json", ROOT / "SEALED_PUBLIC_TEST_SEAL.json",
        ROOT / "gates/validation-v1.json", ROOT / "gates/sealed-public-v1.json",
    ]
    if private.exists() or any((REPO_ROOT / path).exists() for path in targets):
        raise RuntimeError("V5 freeze refuses overwrite")
    private.mkdir(parents=True)
    for target in targets:
        (REPO_ROOT / target).parent.mkdir(parents=True, exist_ok=True)
    protocol = protocol_configuration()
    protocol["split_generator_source_paths"] = [path.as_posix() for path in SPLIT_SOURCES]
    protocol["split_generator_source_bundle_sha256"] = source_bundle_sha256(REPO_ROOT, SPLIT_SOURCES)
    (REPO_ROOT / targets[0]).write_bytes(canonical_json_bytes(protocol))
    created = {}
    for split, seal_name, archive_name, manifest_name in (
        ("validation", "VALIDATION_SEAL.json", "validation-fixtures.zip", "validation-private-manifest.json"),
        ("sealed_public", "SEALED_PUBLIC_TEST_SEAL.json", "sealed-public-fixtures.zip", "sealed-public-private-manifest.json"),
    ):
        scenes = build_split(split)
        archive = private / archive_name
        manifest = private / manifest_name
        private_manifest = save_sealed_archive(scenes, archive)
        manifest.write_bytes(canonical_json_bytes(private_manifest))
        common = {
            "task": TASK, "revision": REVISION, "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "synthetic_only": True, "private_or_article_images": False, "chandler_included": False,
            "generalization_label_included": False, "predecessor_fixture_bytes_reused": False,
            "protocol_path": targets[0].as_posix(), "protocol_sha256": sha256_file(REPO_ROOT / targets[0]),
            "production_approval": False, "release_eligible": False,
        }
        seal = {
            "schema": f"graphreader.ocr-production-composition-{'validation' if split == 'validation' else 'sealed-test'}-seal.v5",
            **common, **proposal_summary(scenes), "split_fingerprint": split_fingerprint(scenes),
            "fixture_archive_path": (PRIVATE_ROOT / archive_name).as_posix(),
            "fixture_archive_sha256": sha256_file(archive),
            "private_manifest_path": (PRIVATE_ROOT / manifest_name).as_posix(),
            "private_manifest_sha256": sha256_file(manifest),
        }
        if split == "validation":
            seal["validation_model_execution_count"] = 0
        else:
            seal |= {
                "truth_hidden_from_model_execution_until_gate": True,
                "prior_public_sample_or_pixel_inspection_used": False,
            }
        (REPO_ROOT / ROOT / seal_name).write_bytes(canonical_json_bytes(seal))
        created[split] = (seal, manifest)
    validation_seal, validation_manifest = created["validation"]
    public_seal, public_manifest = created["sealed_public"]
    candidate_keys = [
        "detector_onnx_sha256", "official_recognizer_onnx_sha256", "numeric_recognizer_onnx_sha256",
        "ambiguity_recognizer_onnx_sha256", "spacing_source_sha256",
    ]
    validation_config = {
        "schema": "graphreader.ocr-production-composition-validation-gate-config.v5",
        "task": TASK, "revision": VALIDATION_REVISION, "expected_candidate_hash_keys": candidate_keys,
        "validation_seal_path": (ROOT / "VALIDATION_SEAL.json").as_posix(),
        "validation_seal_sha256": sha256_file(REPO_ROOT / ROOT / "VALIDATION_SEAL.json"),
        "expected_dataset_manifest_sha256": sha256_file(validation_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, VALIDATION_SOURCES),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(VALIDATION_GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / ROOT / "gates/validation-v1.json").write_bytes(canonical_json_bytes(validation_config))
    public_config = {
        "schema": "graphreader.ocr-production-composition-gate-config.v5",
        "task": TASK, "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": candidate_keys + ["validation_report_sha256"],
        "sealed_public_test_seal_path": (ROOT / "SEALED_PUBLIC_TEST_SEAL.json").as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(REPO_ROOT / ROOT / "SEALED_PUBLIC_TEST_SEAL.json"),
        "expected_dataset_manifest_sha256": sha256_file(public_manifest),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, EVALUATOR_SOURCE_PATHS),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1, "production_approval": False, "release_eligible": False,
    }
    (REPO_ROOT / ROOT / "gates/sealed-public-v1.json").write_bytes(canonical_json_bytes(public_config))
    return {path.name: sha256_file(REPO_ROOT / path) for path in targets}


if __name__ == "__main__":
    print(json.dumps(freeze_split(), indent=2, sort_keys=True))
