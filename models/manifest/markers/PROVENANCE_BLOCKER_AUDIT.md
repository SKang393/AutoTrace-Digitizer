<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Marker-center model provenance and release audit

Audit date: 2026-08-03

## Decision

Exact trained ONNX bytes exist and have a schema-valid manifest. The model is
ignored, unbundled, and not release-eligible. Session 10 acceptance is `FAIL`
and model acceptance is `FAIL`.

The frozen held-out report predates the raw-mask max-gating parity repair. It
is preserved as historical artifact-head-only postprocessing evidence. It is
not production-runtime acceptance evidence, and its metrics were not altered
or rerun.

## Exact model identity

| Field | Value |
|---|---|
| Model ID | `graph-marker-center` |
| Version | `0.1.0` |
| Artifact | `ml/markers/center/artifacts/session10-final/marker-center.onnx` |
| Size | `37,542` bytes |
| SHA-256 | `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a` |
| ONNX opset | `18` |
| Checkpoint SHA-256 | `72471156daec005b7617916a3d81c3966c0a5e36cb93c6c88aaa58f4d1272fb9` |
| Dataset manifest SHA-256 | `f313278c0fa987ec2b9225b3dbdf3ed218fbe8f5c8d6d32aa339ccad2dc4a6c7` |
| Frozen benchmark SHA-256 | `1fb8a69b17163f044e6108dd318e076ad7fa9f16c1a78fe83f6996ce8399e7a7` |

Generated weights and evidence remain under ignored paths and are not
committed.

## Canonical source identity

Training revision is `marker-center-pytorch-v1`. The canonical source digest
is `a8cfd538b164d0e7a3c19c262daf55413eaea9a7639f3a1bc87d0c341b9734ac`.

Stable algorithm:

1. Read these eight files as text: `__init__.py`, `benchmark.py`, `dataset.py`,
   `export.py`, `metrics.py`, `model.py`, `postprocess.py`, and `train.py`.
2. Normalize every CRLF sequence to LF.
3. SHA-256 each normalized UTF-8 file.
4. Sort by file name.
5. Build UTF-8 material with one `name=sha` plus LF line per file.
6. SHA-256 that material.

This digest is stable across CRLF and LF checkouts. It replaces the earlier
noncanonical path-based digest.

## Training and data scope

- Architecture: `compact-pplcnet-depthwise-fpn-v1`.
- Seed: `20260803`.
- Training: 90 epochs and three declared validation threshold comparisons.
- Selected threshold: `0.36`.
- Held-out evaluations during training and model selection: `0`.
- Input-family and degradation-family identifiers are disjoint across train,
  validation, and test.
- Geometry and layout templates repeat across splits.
- No novel-layout or real-data generalization is claimed.
- The dataset manifest records seeds, input and target checksums, center
  counts, degradation identifiers, and hard-negative classes.
- No pretrained weights, external graph figures, article data, private data,
  human annotations, or third-party source were used.

## Deterministic no-heldout reproduction

Ignored evidence:
`ml/markers/center/artifacts/session10-mask-parity-proof/evidence-index.json`.

- Evidence SHA-256:
  `91191738ffaee38c145f162c3a619305e1336f0646c85bc3bc2755f5225ee87c`.
- `heldout_benchmark_invoked`: `false`.
- `heldout_test_evaluations`: `0`.
- Reproduced checkpoint SHA-256:
  `72471156daec005b7617916a3d81c3966c0a5e36cb93c6c88aaa58f4d1272fb9`.
- Reproduced ONNX SHA-256:
  `061a496167382d1bd11bb580bed383d2d1725da2001f9c440b7f1acc59ac116a`.
- Reproduced dataset-manifest SHA-256:
  `f313278c0fa987ec2b9225b3dbdf3ed218fbe8f5c8d6d32aa339ccad2dc4a6c7`.
- ONNX checker passed; CPU parity maximum difference remained
  `2.384185791015625e-06`, below `1e-05`.

The reproduction proves deterministic model and dataset bytes after the
postprocessing parity repair without opening held-out data.

## Tensor and postprocessing contract

- Input `image_and_masks`: float32 NCHW `[N,3,H,W]`.
- Input channels: ink probability, raw text mask, raw artifact mask.
- Output `marker_heads`: activated float32 NCHW `[N,3,H,W]`.
- Output channels: center probability, radius pixels, model artifact
  probability.
- Output stride: one in `model_tensor` coordinates.

At each candidate center, production postprocessing computes:

```text
effective_artifact = max(
    model_artifact_probability,
    sampled_raw_text_mask,
    sampled_raw_artifact_mask)
```

The candidate is rejected when `effective_artifact >= 0.35`. Regression
evidence confirms a raw text or artifact mask value of `0.4` suppresses an
otherwise identical candidate.

Other frozen rules are center threshold `0.36`, 9 by 9 local maxima, minimum
radius `2.5` tensor pixels, minimum NMS distance `5` tensor pixels, and radius
suppression scale `1.25`.

Tensor centers use half-pixel mapping:

```text
frame_x = crop_left + ((tensor_x + 0.5) * crop_width / W) - 0.5
frame_y = crop_top  + ((tensor_y + 0.5) * crop_height / H) - 0.5
```

The recorded frame-to-original inverse produces public `original_pixels`.
Original and enhanced results use minimum-cost maximum one-to-one consensus
within 5 original pixels.

## Provider evidence

Python ONNX Runtime 1.27.0 CPU parity passed. The production
GraphReader.Inference path executed the exact model with CPU and DirectML on
`[1,3,128,128]`. Each returned 49,152 floats.

- CPU inference: `5.1175` ms.
- DirectML inference: `601.636` ms.
- Maximum difference: `1.430511474609375e-06` within `1e-04`.
- Provider evidence SHA-256:
  `f6f8d641f58284fe07d7c191df9d72fae8353cf24e5b8d923621ef1249503e01`.

DirectML succeeded, but ONNX Runtime assigned some shape-related nodes to CPU.
This is a real mixed-node DirectML run, not exclusive GPU execution.

## Historical held-out evidence

Frozen artifact:
`ml/markers/center/artifacts/session10-final/benchmark.json`.

The report is historical artifact-head-only postprocessing evidence from
before raw-mask max-gating parity. It was not rerun after the repair and does
not establish production-runtime acceptance.

Unmodified historical metrics:

- 38 true centers, 38 true positives, 1 false positive, 0 false negatives;
- F1 `0.9870129870129869` at 3 px and 5 px;
- duplicate rate `0`;
- zero standard-mask hits for the eight recorded hard-negative kinds;
- exact point counts in 5 of 6 fixtures;
- zero-mask diagnostic F1 `0.9620253164556963` with 3 false positives,
  including 2 legend hits.

The historical report status is `fail`, and `release_eligible` remains
`false`.

## License and privacy review

Model source, procedural data, checkpoint, and ONNX weights are Copyright 2026
Sungwoo Kang and Apache-2.0. The exact model notice is
`MARKER_CENTER_MODEL_NOTICE.md`; the complete license text is root `LICENSE`.

Dependency license audit evidence:
`ml/markers/center/artifacts/session10-license-repair/evidence-index.json`.

- Evidence SHA-256:
  `ea1faec18b744113b18d3e60815b29ce65cc057062611ed26f6bb9da85368228`.
- Six dependency ledger rows and seven exact wheel-extracted license or notice
  artifacts matched repository file hash, wheel-entry hash, and ledger hash.
- Model and frozen benchmark hashes remained unchanged.

Training dependencies remain unbundled permissive tools. No generated model
weights or private data were added to the manifest tree.

## Release blockers

1. Session 10 acceptance and model acceptance remain `FAIL`.
2. The historical 5-of-6 exact fixture result is not production-runtime
   evidence after the parity repair.
3. No authorized held-out rerun exists for raw-mask max-gated production
   postprocessing.
4. Source must be bound to a tracked commit before release integration.
5. Installer and portable discovery, checksum, CPU fallback, and DirectML
   verification from identical packaged bytes remain required.

## Evidence locations

- Manifest: `models/manifest/markers/graph-marker-center-0.1.0.json`.
- Notice: `models/manifest/markers/MARKER_CENTER_MODEL_NOTICE.md`.
- No-heldout reproduction:
  `ml/markers/center/artifacts/session10-mask-parity-proof/evidence-index.json`.
- License repair:
  `ml/markers/center/artifacts/session10-license-repair/evidence-index.json`.
- Provider probe:
  `tests/GraphReader.Markers.Tests/Detection/TestResults/session10-final-provider-probe.json`.
- Historical held-out artifact:
  `ml/markers/center/artifacts/session10-final/benchmark.json`.
