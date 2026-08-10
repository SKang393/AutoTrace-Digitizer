# Marker-center radial-feature defect class

This revision follows the exhausted dense, candidate-level, and line-aware
marker-center revisions. It reuses no prior weight, candidate ID, threshold,
renderer family, degradation family, or public fixture.

P1 makes one architecture change. It replaces learned convolutional patch
features with 36 fixed radial and topology projections over the existing ink,
text-mask, and artifact-mask planes, then trains a small MLP for marker
probability, sub-grid offset, and radius. The proposal, coordinate, exclusion,
and exact-scene contracts remain fixed.

Training and selection use only new procedural families. The truth-hidden
16-scene public archive is frozen before optimizer execution. Chandler and
private article images are excluded. P1 may open that archive only after every
selection scene and CPU ONNX parity gate passes. A selection or public pass
does not approve production without the independent artifact-mask gate,
production adapter, manifest, model store, notices, packaging, and clean-machine
evidence.
