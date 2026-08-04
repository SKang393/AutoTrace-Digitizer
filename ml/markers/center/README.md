# Marker center detector

This directory trains and evaluates an original compact high-resolution neural
detector for plotted marker centers. It uses only deterministic procedural
project data. No published figures, private data, downloaded weights, or
third-party source are used.

## Architecture and heads

`CompactCenterNet` is a trainable PyTorch model with a PP-LCNet/MobileNetV3-
style depthwise-separable encoder at strides 1, 2, 4, and 8. A compact FPN/
U-Net decoder restores stride-1 resolution. One final tensor contains activated
center probability, radius in tensor pixels, and artifact probability heads.

## Tensor contract

- Input name: `image_and_masks`
- Input: float32 NCHW `[N,3,H,W]`, dynamic N/H/W, range `[0,1]`
- Input channels: ink probability (`1-luminance`), text mask, artifact mask
- Output name: `marker_heads`
- Output: activated float32 NCHW `[N,3,H,W]`
- Output channels: center probability, radius pixels, artifact probability
- Output stride: 1
- Coordinate space: model tensor pixels

At every local-maximum candidate, Python and C# sample the raw text and artifact
masks and take their maximum with the model artifact head. A value at or above
the 0.35 artifact threshold rejects the candidate. The runtime inverse for a grid point is
`cropLeft + ((x + 0.5) * cropWidth / W) - 0.5`, likewise for y, followed by
the recorded frame-to-original inverse. Python and C# use the same 9x9 local
maximum, tie, ordering, radius clamp, and radius-aware NMS rules.

## Data and experiment controls

Train, validation, and test input-family and degradation-family identifiers are
disjoint. Target center geometries and layout templates intentionally repeat
across splits. These procedural results do not establish generalization to new
target geometries, new layout templates, or real articles. A canonical manifest
records every seed and input/target SHA-256 and is sealed before training.
Training rotates through full masks, text-mask dropout,
artifact-mask dropout, and both-mask dropout. Validation and the single final
held-out benchmark both report a zero-mask ablation so hard-negative results
are not accepted solely because the masks reveal the labels.

Model selection compares exactly three thresholds on per-scene aggregated
validation metrics. The test split is opened once after model selection. Center
matching uses maximum-cardinality one-to-one bipartite matching.

## Commands

```powershell
python -m pytest ml/markers/center/tests -q
python -m ml.markers.center.train --output ml/markers/center/artifacts/session10-final
python -m ml.markers.center.export --checkpoint ml/markers/center/artifacts/session10-final/marker-center.pt --output ml/markers/center/artifacts/session10-final/marker-center.onnx --report ml/markers/center/artifacts/session10-final/onnx-parity.json
python -m ml.markers.center.benchmark --checkpoint ml/markers/center/artifacts/session10-final/marker-center.pt --onnx ml/markers/center/artifacts/session10-final/marker-center.onnx --output ml/markers/center/artifacts/session10-final/benchmark.json
```

Production repair revision status:

The `marker-center-production-repair-v1` budget is exhausted. Its retained CLI
refuses before creating output. A future authorized repair requires a new
preregistered revision and cannot reuse `P1`, `P2`, or `P3`.

The 2026-08-04 three-candidate budget is invalid because executed mask-channel
dropout was not preregistered. It also failed the independent confirmation
exact-count gate. The candidates remain rejected and the manifest remains
fail-closed. The corrected future implementation preserves both mask channels
and records global and model-initialization seeds, but the exhausted candidates
are not rerun.

Direct wheel artifacts are hash-locked in `requirements*.txt`. They can be
verified against the ignored audit cache with `pip install --dry-run --no-deps
--require-hashes --no-index --find-links ml/markers/center/artifacts/provenance
-r ml/markers/center/requirements-test.txt`.

## License and release status

Project source, procedural training data, and trained model weights are
Copyright 2026 Sungwoo Kang and Apache-2.0. Training dependencies are
unbundled permissive tooling recorded in `DEPENDENCY_PROVENANCE.csv`. The exact
ONNX SHA-256, CPU provider parity, tensor contract, and benchmark must be copied
into the separately owned model manifest before release eligibility is claimed.
