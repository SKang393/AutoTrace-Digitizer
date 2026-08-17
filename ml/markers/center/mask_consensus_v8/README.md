# Marker-center mask-consensus V8

V8 is a fresh marker-center and artifact-mask defect class. Its design uses
only the aggregate terminal V7 result: 50 of 96 exact selection scenes, 11
false positives, 85 false negatives, four arrowhead hits, one marker-artifact
hit, artifact precision `0.8523057467207811`, artifact recall
`0.9580230017289313`, and CPU ONNX parity
`0.000011861324310302734`. No V7 case detail, pixels, fixture identity,
checkpoint, Chandler image, private/article image, or external dataset is used.

Before any training or model execution, V8 freezes 512 training, 128 visible
validation, and 160 truth-hidden public procedural scenes. Renderer,
degradation, scene-ID, and seed families are disjoint. Every split must prove
zero truth/prohibited acceptance overlap, zero truth/mask conflict, and that
every prohibited hard point is present in artifact truth.

The split is now frozen with zero model executions and zero optimizer steps.
Train, validation, and sealed-public archive SHA-256 values are
`17a81be5...42d7`, `533b32a7...8412`, and `0f39a5ae...4002`. The tracked
selection manifest, split-freeze report, and public seal SHA-256 values are
`48f888ac...1a32`, `79263032...4938`, and `d1376f99...00b0`. The exact P1
runner and single-use public evaluator were preregistered in commit
`4e20674d0d7a15896005a066c2054753dbf5d7dd`, tree
`0e51075b8cd082b9ce48e0232fa008fee9e9627a`. The canonical ledger now
authorizes only P1 for one execution from those exact committed identities.

P1 changes one defect class: model capacity and explicit mask consensus. A new
full-resolution residual U-Net predicts missing artifact structure, retains the
input artifact seed by a max operation, and suppresses center probability with
the exact text and composed artifact planes. It preserves the production
`[N,3,H,W]` input and output contract, two-pass artifact composition, frozen
postprocessing, thresholds, artifact threshold, zero-error scene gates, and
`1e-5` CPU parity limit.

P1 executed exactly once for 3,584 optimizer steps and failed closed. It passed
122 of 128 visible scenes with 1,193 true positives, six false positives, 23
false negatives, zero duplicates, zero prohibited-structure hits, and zero
marker-artifact hits. Artifact precision was `0.7864128234384776`, artifact
recall was `0.9843701768692439`, and CPU ONNX parity was
`0.00002002716064453125`, above the fixed `1e-5` limit. Candidate report,
checkpoint, and ONNX SHA-256 values are `223201e3...0f7e`,
`46eb64a9...122`, and `13b6204b...9f8c`. The public archive remained unopened
with zero evaluations, and no manifest, store, package, private validation,
approval, or release state changed.

P2 reused the exact consumed P1 checkpoint and ONNX with zero optimizer steps
and changed only the artifact decision threshold from `0.35` to `0.45`. It
still passed only 122 of 128 visible scenes, with 1,193 true positives, six
false positives, 23 misses, artifact precision `0.7872885671271697`, artifact
recall `0.9829926331356857`, and failed CPU parity
`0.00001621246337890625`. Its report and result seal SHA-256 values are
`14d02a58...af9` and `15b3b8c8...cb3`.

Final P3 acquired its single-use committed authorization, then failed closed
before training during the exact P2 per-channel parity-reproduction preflight.
Center-channel parity reproduced as `0.000003874301910400391` instead of the
preregistered `0.0000033080577850341797`; radius and artifact values matched
exactly. Zero optimizer steps ran. Candidate report, tracked result, opened
seal, and result seal SHA-256 values are `a40cbdee...9f9b`,
`fd2dfa1a...04f8`, `ca153280...47b4`, and `e676945f...c91d`. P1, P2, and P3
are consumed, V8 is exhausted, and it cannot rerun. The truth-hidden public
archive remains unopened at zero evaluations. No manifest, model store,
packaging, private Chandler validation, production approval, or release is
authorized. A future attempt requires a new preregistered defect class and
fresh splits.
