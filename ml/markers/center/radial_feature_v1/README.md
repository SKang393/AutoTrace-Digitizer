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

P3 is consumed and selected as the final candidate in this defect-class budget.
It performed zero optimizer steps and reused the exact best P1 checkpoint and
ONNX. The isolated postprocessing change searches only the deterministic
one-pixel neighborhood when the existing discrete geometry probe rejects a
regressed center. P3 passed all 9 frozen selection scenes and all 63 markers at
threshold `0.3` with zero false positives, misses, duplicates, or prohibited
hits. CPU ONNX parity remained `1.9073486328125e-06`. Its report SHA-256 is
`67b5ea3b28973f0bd24ae0f755713af1c70b6fe6a9b2437268be5975b9f14af3`.
The single authorized truth-hidden public evaluation is consumed and failed.
P3 passed 11 of 16 scenes exactly, retained 109 of 116 markers, and missed seven
markers across five scenes. It produced zero false positives, duplicates, and
prohibited-structure hits. The exact evaluator output is tracked at
`PUBLIC_GATE_REPORT.json` with SHA-256
`e343bb72cf098c9a7598d91a0264288eadd7401967f7197f5d5fe6249be58996`.
All three candidates and the public-gate budget are exhausted. This revision
cannot rerun or approve a production marker-center payload.

Training and selection use only new procedural families. The truth-hidden
16-scene public archive is frozen before optimizer execution. Chandler and
private article images are excluded. P3 may open that archive only after every
selection scene and CPU ONNX parity gate passes. A selection or public pass
does not approve production without the independent artifact-mask gate,
production adapter, manifest, model store, notices, packaging, and clean-machine
evidence.
