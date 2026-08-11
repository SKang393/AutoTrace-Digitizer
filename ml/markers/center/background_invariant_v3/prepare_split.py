# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sungwoo Kang
"""Freeze fresh marker background-invariance selection and public evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets

from ml.markers.center.background_invariant_v3.dataset import (
    DATASET_REVISION,
    DEGRADATIONS,
    SEALED_PUBLIC_FAMILIES,
    build_sealed_public_scenes,
    save_sealed_public_archive,
    selection_manifest,
)
from ml.markers.center.background_invariant_v3.public_gate import (
    EVALUATOR_SOURCE_PATHS,
    GATE_CONFIG,
    REVISION as PUBLIC_REVISION,
)
from ml.markers.gate_seal import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    source_bundle_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRIVATE_ROOT = Path("ml/markers/center/artifacts/background-invariant-v3/split-freeze")
DEFAULT_SELECTION_MANIFEST = Path(
    "ml/markers/center/background_invariant_v3/SELECTION_MANIFEST.json"
)
DEFAULT_SEALED_TEST_SEAL = Path(
    "ml/markers/center/background_invariant_v3/SEALED_PUBLIC_TEST_SEAL.json"
)
DEFAULT_GATE_CONFIG = Path(
    "ml/markers/center/background_invariant_v3/gates/sealed-public-p1.json"
)
DEFAULT_TRAINING_CONFIG = Path(
    "ml/markers/center/background_invariant_v3/training/p1.json"
)
SOURCE_PATHS = (
    Path("ml/markers/center/background_invariant_v3/dataset.py"),
    Path("ml/markers/center/background_invariant_v3/prepare_split.py"),
)
RUNNER_SOURCE_PATHS = (
    Path("ml/markers/center/background_invariant_v3/dataset.py"),
    Path("ml/markers/center/background_invariant_v3/pipeline.py"),
    Path("ml/markers/center/background_invariant_v3/candidate_runner.py"),
    Path("ml/markers/center/background_invariant_v3/public_gate.py"),
    Path("ml/markers/center/runtime_consistency_v2/candidate_runner.py"),
    Path("ml/markers/center/runtime_consistency_v2/pipeline_p2.py"),
    Path("ml/markers/center/radial_feature_v1/model.py"),
    Path("ml/markers/center/radial_feature_v1/pipeline_p3.py"),
    Path("ml/markers/center/line_aware_v1/model.py"),
    Path("ml/markers/center/line_aware_v1/pipeline.py"),
    Path("ml/markers/gate_seal.py"),
    Path("ml/markers/training_budget.py"),
)


def freeze_split(
    *,
    private_root: Path,
    selection_manifest_path: Path,
    sealed_test_seal_path: Path,
    gate_config_path: Path,
    training_config_path: Path,
) -> dict[str, object]:
    private = REPO_ROOT / private_root
    selection_path = REPO_ROOT / selection_manifest_path
    seal_path = REPO_ROOT / sealed_test_seal_path
    gate_path = REPO_ROOT / gate_config_path
    training_path = REPO_ROOT / training_config_path
    seed_path = private / "private-seed.json"
    archive_path = private / "sealed-public-fixtures.npz"
    private_manifest_path = private / "sealed-public-private-manifest.json"
    targets = (
        seed_path,
        archive_path,
        private_manifest_path,
        selection_path,
        seal_path,
        gate_path,
        training_path,
    )
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RuntimeError("Split freeze refuses to overwrite existing evidence: " + ", ".join(existing))
    private.mkdir(parents=True, exist_ok=False)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    training_path.parent.mkdir(parents=True, exist_ok=True)

    secret_seed = secrets.randbelow(2_000_000_000) + 100_000
    seed_path.write_bytes(
        canonical_json_bytes(
            {
                "schema": "graphreader.marker-center-background-invariant-private-seed.v3",
                "dataset_revision": DATASET_REVISION,
                "seed": secret_seed,
            }
        )
    )
    selection_path.write_bytes(canonical_json_bytes(selection_manifest()))
    scenes = build_sealed_public_scenes(secret_seed)
    private_manifest = save_sealed_public_archive(scenes, archive_path)
    private_manifest_path.write_bytes(canonical_json_bytes(private_manifest))

    seal = {
        "schema": "graphreader.marker-center-background-invariant-sealed-test-seal.v3",
        "task": "marker-center",
        "revision": "marker-center-background-invariant-v3",
        "dataset_revision": DATASET_REVISION,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scenes),
        "family_ids": list(SEALED_PUBLIC_FAMILIES),
        "degradation_ids": list(DEGRADATIONS["sealed_public"]),
        "synthetic_only": True,
        "private_or_article_images": False,
        "chandler_included": False,
        "prior_exposed_fixture_bytes_reused": False,
        "truth_hidden_from_candidate_until_selection_pass": True,
        "fixture_archive_path": private_root.joinpath("sealed-public-fixtures.npz").as_posix(),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_path": private_root.joinpath(
            "sealed-public-private-manifest.json"
        ).as_posix(),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "private_seed_sha256": sha256_file(seed_path),
        "selection_manifest_path": selection_manifest_path.as_posix(),
        "selection_manifest_sha256": sha256_file(selection_path),
        "split_generator_source_paths": [path.as_posix() for path in SOURCE_PATHS],
        "split_generator_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, SOURCE_PATHS),
        "public_release_eligible": False,
    }
    seal_path.write_bytes(canonical_json_bytes(seal))

    gate = {
        "schema": "graphreader.marker-center-background-invariant-gate-config.v3",
        "task": "marker-center",
        "revision": PUBLIC_REVISION,
        "expected_candidate_hash_keys": ["onnx_sha256", "selection_report_sha256"],
        "sealed_public_test_seal_path": sealed_test_seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "expected_dataset_manifest_sha256": sha256_file(private_manifest_path),
        "expected_evaluator_source_bundle_sha256": source_bundle_sha256(
            REPO_ROOT, EVALUATOR_SOURCE_PATHS
        ),
        "expected_gate_config_sha256": sha256_bytes(canonical_json_bytes(GATE_CONFIG)),
        "evaluation_limit": 1,
        "production_approval": False,
        "release_eligible": False,
    }
    gate_path.write_bytes(canonical_json_bytes(gate))

    training = {
        "schema": "graphreader.marker-center-background-invariant-candidate-config.v3",
        "task": "marker-center",
        "revision": "marker-center-background-invariant-v3",
        "candidate_id": "P1",
        "experiment_ordinal": 1,
        "experiment_budget": 1,
        "trigger": "The consumed runtime-consistency P2 public gate failed seven scenes, and five of those seven used smooth background-ramp degradation. Those exposed fixtures are trigger evidence only and cannot be rerun or used for selection.",
        "isolated_change": "subtract the lower-middle median from only the ink channel of each 33 by 33 proposal patch before the exact unchanged ONNX; retain masks, proposal eligibility, threshold, geometry refinement, duplicate suppression, payload bytes, and all gates",
        "source_result_path": "ml/markers/center/runtime_consistency_v2/P2_RESULT.json",
        "source_result_sha256": "451909449574a0d0e02fe5e0460667f806c41963f88f14c05de11a9cfbaff999",
        "source_training_report_path": "ml/markers/center/artifacts/runtime-consistency-v2/P2-run/candidate-report.json",
        "source_training_report_sha256": "010b43518117e0d0013e50b72f81d69052635e494d6a27187746354f5995d904",
        "source_checkpoint_path": "ml/markers/center/artifacts/runtime-consistency-v2/P2-run/marker-center-runtime-consistency-p2.pt",
        "source_checkpoint_sha256": "6b670a6f29454d7f63527f57210aa918540a817fca156a71b96872ff09aa2787",
        "source_onnx_path": "ml/markers/center/artifacts/runtime-consistency-v2/P2-run/marker-center-runtime-consistency-p2.onnx",
        "source_onnx_sha256": "924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b",
        "source_model_seed": 20261001,
        "optimizer_steps": 0,
        "weights_changed": False,
        "preprocess_revision": "patch-ink-median-background-subtraction-v1",
        "background_statistic": "torch.median over each 33x33 ink patch",
        "postprocess_revision": "radial-local-consensus-calibration-v2",
        "selected_threshold": 0.25,
        "minimum_center_separation": 6.5,
        "onnx_parity_tolerance": 0.00001,
        "selection_manifest_path": selection_manifest_path.as_posix(),
        "selection_manifest_sha256": sha256_file(selection_path),
        "sealed_public_test_seal_path": sealed_test_seal_path.as_posix(),
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "expected_runner_source_bundle_sha256": source_bundle_sha256(REPO_ROOT, RUNNER_SOURCE_PATHS),
        "public_gate_authorized_on_selection_pass": True,
        "public_gate_evaluations": 0,
        "private_or_article_images": False,
        "chandler_included": False,
        "production_approval": False,
        "release_eligible": False,
    }
    training_path.write_bytes(canonical_json_bytes(training))
    return {
        "scene_count": len(scenes),
        "selection_manifest_sha256": sha256_file(selection_path),
        "sealed_public_test_seal_sha256": sha256_file(seal_path),
        "fixture_archive_sha256": sha256_file(archive_path),
        "private_manifest_sha256": sha256_file(private_manifest_path),
        "gate_config_sha256": sha256_file(gate_path),
        "training_config_sha256": sha256_file(training_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", type=Path, default=DEFAULT_PRIVATE_ROOT)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--sealed-test-seal", type=Path, default=DEFAULT_SEALED_TEST_SEAL)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    args = parser.parse_args()
    result = freeze_split(
        private_root=args.private_root,
        selection_manifest_path=args.selection_manifest,
        sealed_test_seal_path=args.sealed_test_seal,
        gate_config_path=args.gate_config,
        training_config_path=args.training_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
