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

P2 is the only currently preregistered candidate. It changes one training
coefficient: the positive center-offset loss weight increases from `1.5` to
`3.0`. It uses a new deterministic seed and new weights while retaining P1's
architecture, frozen data, proposal and postprocessing code, thresholds,
optimizer, and epoch count. P2 cannot execute until its configuration, runner
source bundle, P1 result seals, and canonical budget authorization are
committed together.

Training and selection use only new procedural families. The truth-hidden
16-scene public archive is frozen before optimizer execution. Chandler and
private article images are excluded. P2 may open that archive only after every
selection scene and CPU ONNX parity gate passes. A selection or public pass
does not approve production without the independent artifact-mask gate,
production adapter, manifest, model store, notices, packaging, and clean-machine
evidence.
