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

P1 is preregistered but not authorized in this checkpoint. No optimizer step,
candidate selection inference, public evaluation, manifest, model-store promotion, packaging,
private validation, or production approval has occurred. Execution requires a
separate commit that binds the committed preregistration tree and authorizes
only P1. The public archive remains locked even if visible selection later
passes, until a further candidate-specific public-gate authorization is
committed.
