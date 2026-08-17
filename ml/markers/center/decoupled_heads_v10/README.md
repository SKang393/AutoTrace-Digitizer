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

P2 preregistration commit `898d46ce99acfe9c24ef6e55f5af7aaac36fea6b`
and tree `3ebfbfa9affddbdcec4ff0f63d9388c9a94e8120` bind its exact
configuration and runner bytes. P2 executed once from the separate
authorization commit and is consumed. It completed all 1,792 optimizer steps
and passed CPU ONNX parity at `8.881092071533203e-06`. It failed fixed visible
selection with 121 of 128 exact scenes, 3 false positives, 28 false negatives,
one marker-artifact hit, artifact precision `0.7819652572390059`, artifact
recall `0.9677843660304165`, and no passing threshold window. Report,
checkpoint, and ONNX SHA-256 values are
`a9ac118fe55c801941ad558b02f8d2a75c9f506023ae394ad903a723418b443b`,
`e64c6bf08ccc3f7e60bcf48710230d3081c3cd8f44d0a474d8e303224bf74a40`,
and `bce86a20624802539b428a349e13710c88e8e8c6329175f758ed846023435b24`.
Only aggregate metrics were inspected.

P3 is the final preregistered V10 candidate. It retains the exact P2
architecture, optimizer, seed, frozen split, four-way reflection schedule,
loss form, and 1,792-step budget while changing only the artifact Tversky
false-positive and false-negative balance from `0.80/0.20` to `0.95/0.05`.
This targets P2's aggregate artifact-precision and marker-masking failures
without using fixture detail or pixels. It retrains from scratch, reuses no P2
checkpoint, and remains blocked until a separate commit binds and authorizes
its exact bytes once. The fixed selection thresholds and zero-error gates are
unchanged.

No V10 payload is approved, stored, packaged, privately validated, or release
eligible.
