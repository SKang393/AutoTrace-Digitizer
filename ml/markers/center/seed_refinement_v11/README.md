# Marker-center seed refinement V11

V10 is terminal. All three candidates failed visible selection, and its tracked
aggregate feasibility proof shows that `maximum(seed, learned_artifact)` can
reach at most `0.796366958474284` artifact precision against the unchanged
`0.90` gate. More V10 loss or threshold tuning cannot remove seed false
positives.

V11 is a new three-candidate defect class. P1 keeps the independent
full-resolution center and artifact towers, scratch training, exact four-way
reflection schedule, P3 precision-weighted artifact loss, fixed 2.5-pixel
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

P2 is preregistered from that aggregate-only result. It reuses no P1 model
bytes and retrains the exact architecture, frozen split, optimizer, seed,
four-way reflection schedule, existing losses, 1,792-step budget, thresholds,
CPU provider, parity limit, and zero-error gates from scratch. Its only change
adds a weight-3.0 binary-cross-entropy term on artifact pixels that are both
seed-negative and truth-negative. This penalizes unsupported seed additions
without penalizing additions supported by training truth. P2 is authorized
exactly once from preregistration commit
`589e8892718feaa011849132313c1eb6e71f534e` and tree
`8b70c750ca256de7ba5d5a04bf7408089afb4e52`. The single-use training seal
must refuse a rerun. The public archive remains separately locked even if P2
visible selection later passes.
