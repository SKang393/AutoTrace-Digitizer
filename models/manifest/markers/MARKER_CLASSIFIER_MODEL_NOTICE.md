<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Sungwoo Kang
-->

# Graph marker classifier model notice

Model ID: `graph-marker-classifier`

Model version: `0.1.0`

Artifact: `marker-classifier-probability-packed.onnx`

SHA-256:
`26f9304f1689053a0b94aa896a1e239f6ade1e5c1920736a3535c1b32f803b8a`

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

Runtime-repair P1 preserves the exact selected checkpoint with zero optimizer
steps and exposes calibrated probabilities instead of high-magnitude logits.
Generated weights remain ignored and are not committed.

The once-only public-v3 and disjoint confirmation-v3 procedural gates passed
the fixed shape, fill, artifact, minority-shape, and `1e-5` CPU ONNX parity
thresholds. The exact payload executes through the production C# probability
decoder on CPU and DirectML with maximum provider difference
`3.5762786865234375e-07`. This notice does not waive production model-store,
end-to-end workflow, private-graph, or clean-machine gates. The exact payload,
notice, benchmark evidence, package index, CPU and DirectML providers, and
production resolver now pass together, so this classifier payload is approved
for production discovery and packaging. It is not application release approval.
