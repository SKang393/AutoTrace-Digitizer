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

P1 executed once and is consumed. It completed all 1,792 optimizer steps and
passed CPU ONNX parity at `9.894371032714844e-06`, with zero duplicates,
prohibited-structure hits, or marker-artifact hits. It failed fixed visible
selection with 120 of 128 exact scenes, 11 false positives, 23 false negatives,
artifact precision `0.7857918313961029`, artifact recall
`0.9770655093456437`, and no passing threshold window. Only aggregate metrics
were inspected. The truth-hidden public archive remains locked and unopened.

P2 is preregistered but blocked until a later commit binds its exact
preregistration commit and tree and explicitly authorizes it once. P2 retrains
the exact P1 architecture, loss, optimizer, seed, frozen split, and 1,792-step
budget from scratch. Its only change is an exact four-way reflection schedule
applied consistently to every training tensor, target, center, and hard-negative
coordinate. It reuses no P1 checkpoint and performs no interpolation. The fixed
selection thresholds and zero-error gates are unchanged.

No V10 payload is approved, stored, packaged, privately validated, or release
eligible.
