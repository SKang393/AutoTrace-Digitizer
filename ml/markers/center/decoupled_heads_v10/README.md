# Marker-center decoupled heads V10

V9 ended after all three candidates failed visible selection. Its final model
passed CPU ONNX parity and artifact recall, but shared one encoder for center and
artifact objectives while retaining 12 false centers, 18 missed centers, and
artifact precision below the fixed `0.90` gate.

V10 is a distinct defect class. It trains independent full-resolution center
and artifact towers from scratch. The artifact output is detached only from the
center tower's gradient path, not from inference, so ordinary execution still
uses the exact two-pass checksum-bound artifact mask to suppress centers. The
fixed three-channel input, three-head output, center postprocessing, and public
contract remain unchanged.

Before any training, the repository freezes 512 training scenes, 128 visible
selection scenes, and 160 truth-hidden public scenes. Renderer, degradation,
seed, scene identity, truth, prohibited, and artifact families are disjoint.
The split contains procedural data only and excludes Chandler, private or
article images, external datasets, and prior fixture bytes.

P1 is preregistered but blocked until a later commit binds the preregistration
commit and tree and explicitly authorizes it. A passing candidate must produce
exact counts in every visible scene, zero false positives, false negatives,
duplicates, prohibited hits, or marker-artifact hits across three consecutive
thresholds, artifact precision at least `0.90`, artifact recall at least
`0.95`, and CPU ONNX parity at most `1e-5`. The public gate remains separately
locked, one-use, and truth-hidden.

No V10 payload is approved, stored, packaged, privately validated, or release
eligible at preregistration.

