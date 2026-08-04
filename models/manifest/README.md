<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Model manifests

The current production approval decision is recorded in
`PRODUCTION_MODEL_MATRIX.md`. No model is approved solely because a manifest or
candidate checksum exists.

Only metadata conforming to `contracts/model-manifest.schema.json` belongs in
this directory. Model binaries and weights are not part of Goal 00 and are
ignored by Git unless a later session completes the required license,
redistribution, checksum, provenance, and benchmark review.
