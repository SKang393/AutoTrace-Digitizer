<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Graph marker classifier model notice

Model ID: `graph-marker-classifier`

Model version: `0.1.0`

Artifact: `marker-classifier-packed.onnx`

SHA-256:
`59b4af98fe40abd436f01a8c14bf0d12a7c82682ec072c65cef92881aa18b0ef`

Copyright 2026 Sungwoo Kang.

The model architecture, training code, procedurally generated training data,
checkpoint, and exported model are original Graph Auto Reader work licensed
under Apache License 2.0. The complete license text is the repository root
`LICENSE` file.

Training used only deterministic project-generated marker patches. It used no
private study image, article figure, external dataset, pretrained weight, or
downloaded model weight. Training dependencies are unbundled permissive tools
recorded in `ml/markers/classifier/DEPENDENCY_PROVENANCE.csv` and the adjacent
training-tool notices.

The exact checkpoint and packed ONNX were reproduced byte for byte in Goal 19
with seed `20260803`. Generated weights remain ignored and are not committed.

This model is an unapproved candidate. The selected checkpoint passed the
historical single sealed procedural held-out gate, but validation shape
macro-F1 was below the session-local threshold. The packed runtime wrapper has
validation parity and exact CPU and DirectML provider evidence, but it has not
received a direct held-out runtime benchmark, a maintainer-approved numeric
gate, private-graph validation, or packaged model-discovery validation.
