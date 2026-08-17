# Marker-center seed refinement V11

V10 is terminal. All three candidates failed visible selection, and its tracked
aggregate feasibility proof shows that `maximum(seed, learned_artifact)` can
reach at most `0.796366958474284` artifact precision against the unchanged
`0.90` gate. More V10 loss or threshold tuning cannot remove seed false
positives.

V11 is a new three-candidate defect class. P1 keeps the independent
full-resolution center and artifact towers, scratch training, exact four-way
reflection schedule, V10 P3 precision-weighted artifact loss, fixed 2.5-pixel
radius, CPU provider, `1e-5` ONNX parity limit, and all zero-error selection
gates. Its isolated architectural change treats the checksum-bound artifact
seed as a proposal prior. A bounded learned logit correction can add missing
artifact pixels and remove false seed pixels. The refined artifact probability
is consumed in one pass by postprocess profile
`nonmonotonic_seed_refinement_v1`.

Before any model execution, V11 froze 512 training scenes, 128 visible
selection scenes, and 160 truth-hidden public scenes. A V11-only procedural
paper-field degradation makes all 800 source images byte-distinct from V10 as
well as mutually distinct inside V11. Renderer names, degradation names, seed
offsets, scene identities, archives, and source hashes are new. The split is
synthetic only. It contains no Chandler, article, private, external dataset, or
prior fixture bytes.

P1 consumed its single authorized 1,792-step CPU run and failed visible
selection. It passed ONNX parity at `6.377696990966797e-06`, retained zero
false-positive centers, duplicates, marker-artifact hits, or prohibited
structure hits at its selected threshold, and passed artifact recall at
`0.9687703003413912`. It reached 122 of 128 exact scenes with 26 missed
centers, but artifact precision was only `0.7815472274567087` against the
unchanged `0.90` gate. The aggregate refined mask added 250,166 seed-negative
pixels and removed 1,175 seed-positive pixels. No case detail or fixture pixels
were inspected, and the public archive remains unopened at zero evaluations.

P2 consumed its single authorized from-scratch run and failed visible
selection. It passed CPU ONNX parity at `9.566545486450195e-06` and retained
zero false-positive centers, duplicates, marker-artifact hits, or prohibited
structure hits. It still reached only 122 of 128 exact scenes with 29 missed
centers. Artifact precision was `0.782828150056521` and artifact recall was
`0.9603460905861702`. Relative to P1, its targeted seed-addition loss removed
only 5,474 additions and 78 additional seed pixels, while exact-scene count did
not change. The candidate report, checkpoint, ONNX, opening seal, and result
seal SHA-256 values are `b886281d...24ec`, `d6c22bb1...03af`,
`8ad7439d...2a9`, `ff7bca7b...bb7`, and `bcbf3a72...195e`. Only aggregate
evidence was used. No case detail or fixture pixels were inspected.

P3 is the final preregistered candidate. It retrains the exact P2
architecture, frozen split, optimizer, seed, four-way reflection schedule, P2
losses, 1,792-step budget, thresholds, CPU provider, parity limit, and all
zero-error gates from scratch. Its only change adds a weight-6.0
binary-cross-entropy term on artifact pixels that are seed-positive and
truth-negative. This directly trains removal of false seed pixels without
penalizing truth-supported seed pixels. P3 reuses no P2 checkpoint and remains
blocked until this preregistration is committed, pushed, and separately
authorized. The public archive remains separately locked at zero evaluations.
