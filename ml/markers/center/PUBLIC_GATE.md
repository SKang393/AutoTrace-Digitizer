# Marker-center public gate

This gate was frozen before opening the candidate ONNX on 2026-08-04. It uses
only deterministic Apache-2.0 procedural data and is not used for training,
threshold selection, or model selection.

The fixed split contains three new renderer/degradation families and three new
layout templates. Every scene contains all eight prohibited structure kinds.

Approval requires all conditions together:

- the predicted marker count exactly equals truth in every scene;
- there are zero duplicate centers;
- there are zero detections on text, axes, ticks, dividers, brackets,
  arrowheads, legends, or line intersections.

The evaluator requires committed, clean evaluator and split sources. Before
inference it validates the frozen task, revision, and ordered candidate-hash
schema, then atomically creates a repository-scoped seal. The seal binds that
identity, the split hash, gate configuration, and evaluator-source hash. The
same identity is rejected even when a different output directory is requested. A
passing marker-owned gate remains release-ineligible until production discovery
and packaging evidence exist.
