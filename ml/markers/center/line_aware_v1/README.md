# Marker-center line-aware defect class

This revision follows the exhausted dense and candidate-level revisions. It
does not reuse their weights, exposed public fixtures, family IDs, degradation
IDs, thresholds, or candidate IDs.

P1 changed one defect boundary: candidate proposals and regressed centers must
both agree with the text and artifact masks, while a deterministic radial ink
check rejects line-only structure. A dual-branch CNN learns ink and mask
context separately. P1 was consumed when its ONNX preflight failed before the
optimizer was created. It opened no public fixtures, ran no public evaluation,
and produced no checkpoint.

P2 is the only authorized candidate. It preserves P1's inputs, outputs,
training scenes, validation scenes, hyperparameters, thresholds, and geometry.
Its only model change replaces unsupported adaptive 8-to-3 pooling with a fixed
4-by-4 stride-two average pool that preserves the 3-by-3 head contract. P2 uses
a new seed because P1 produced no weights. Training and validation use only new
procedural families.

The truth-hidden public archive is generated once before training and can be
opened once only after a candidate passes every selection scene and CPU ONNX
parity.

No result from selection or the public gate alone authorizes production. A
passing candidate still requires an independent artifact-mask gate, production
adapter execution, a checksum-bound manifest and model store, notices,
packaging discovery, and clean-machine proof.
