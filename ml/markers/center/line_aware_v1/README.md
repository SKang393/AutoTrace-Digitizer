# Marker-center line-aware defect class

This revision follows the exhausted dense and candidate-level revisions. It
does not reuse their weights, exposed public fixtures, family IDs, degradation
IDs, thresholds, or candidate IDs.

P1 changes one defect boundary: candidate proposals and regressed centers must
both agree with the text and artifact masks, while a deterministic radial ink
check rejects line-only structure. A dual-branch CNN learns ink and mask
context separately. Training and validation use only new procedural families.
The truth-hidden public archive is generated once before training and can be
opened once only after a candidate passes every selection scene and CPU ONNX
parity.

No result from selection or the public gate alone authorizes production. A
passing candidate still requires an independent artifact-mask gate, production
adapter execution, a checksum-bound manifest and model store, notices,
packaging discovery, and clean-machine proof.
