# Marker-center radial-feature defect class

This revision follows the exhausted dense, candidate-level, and line-aware
marker-center revisions. It reuses no prior weight, candidate ID, threshold,
renderer family, degradation family, or public fixture.

P1 made one architecture change. It replaced learned convolutional patch
features with 36 fixed radial and topology projections over the existing ink,
text-mask, and artifact-mask planes, then trains a small MLP for marker
probability, sub-grid offset, and radius. The proposal, coordinate, exclusion,
and exact-scene contracts remain fixed.

P1 is consumed and cannot rerun. Its single run produced valid CPU ONNX bytes
with maximum absolute parity error `1.9073486328125e-06`, but it passed only 7
of 9 frozen selection scenes. It retained 61 of 63 markers with zero false
positives, duplicates, or prohibited-structure hits. Both misses had available
high-confidence proposals whose regressed centers failed the unchanged
geometry-consensus filter. The public archive remained closed.

P2 is also consumed and cannot rerun. It changed one training coefficient by
increasing the positive center-offset loss weight from `1.5` to `3.0`, used a
new deterministic seed, and produced CPU ONNX parity error
`9.5367431640625e-07`. It worsened frozen selection to 5 of 9 exact scenes and
59 of 63 retained markers at threshold `0.15`, still with zero false positives,
duplicates, or prohibited hits. Its public archive remained closed.

P3 is the only currently preregistered candidate and the final candidate in
this defect-class budget. It performs zero optimizer steps and reuses the exact
best P1 checkpoint and ONNX. The isolated change is postprocessing: when the
existing discrete geometry probe rejects a regressed center, P3 searches only
the deterministic one-pixel neighborhood for the nearest unmasked position
that satisfies the same geometry consensus. Nonmaximum suppression and every
other contract remain unchanged. Direct validation analysis selected this
defect correction because both P1 misses had valid, high-confidence proposals
and were rejected only by one-pixel probe quantization.

Training and selection use only new procedural families. The truth-hidden
16-scene public archive is frozen before optimizer execution. Chandler and
private article images are excluded. P3 may open that archive only after every
selection scene and CPU ONNX parity gate passes. A selection or public pass
does not approve production without the independent artifact-mask gate,
production adapter, manifest, model store, notices, packaging, and clean-machine
evidence.
