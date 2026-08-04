# Marker-classifier public gate

This gate was frozen before opening the packed runtime ONNX on 2026-08-04. It
uses only deterministic Apache-2.0 procedural patches and is not used for
training, temperature fitting, threshold selection, or model selection.

The fixed split uses two new renderer/degradation families and two new spatial
templates. It includes every shape/fill combination, line-contact and mixed
series context, minority star/asterisk/cross probes, and all eight artifact
kinds.

Approval requires all conditions together:

- shape macro-F1 at least `0.90`;
- fill macro-F1 at least `0.90`;
- artifact F1 exactly `1.0`;
- each minority shape F1 at least `0.90`;
- direct packed ONNX parity at most `1e-5`.

The evaluator requires committed, clean evaluator and split sources. Before
inference it validates the frozen task, revision, and ordered candidate-hash
schema, then atomically creates a repository-scoped seal. The seal binds that
identity, the split hash, gate configuration, and evaluator-source hash. The
same identity is rejected even when a different output directory is requested. A
passing marker-owned gate remains release-ineligible until production discovery
and packaging evidence exist.
