# Marker-center domain-randomized V7

V7 is a fresh marker-center defect class. It uses only the aggregate terminal
V6 public result: 0/64 exact scenes, 190 false positives, 297 misses, four
prohibited hits, artifact precision `0.6145023044357781`, and artifact recall
`0.6914750693739623`. No V6 public fixture bytes, case-level failures, Chandler,
private images, article data, external datasets, or predecessor weights are
used.

Before training, V7 freezes 384 training, 96 visible validation, and 96
truth-hidden public scenes. The train and validation corpus spans independent
layout, marker-shape, mask-imperfection, illumination, blur, compression,
stroke, quantization, scanline, and noise families. The public renderer and
degradation family identities are disjoint and its archive remains unopened.
Every split has zero truth/prohibited acceptance overlap, zero truth/mask
conflict, and every prohibited hard point is present in artifact truth.

P1 preserves the V6 model capacity, three-channel dense input/output contract,
loss, postprocessing, threshold set, exact-count gate, prohibited-structure
gate, artifact precision/recall gate, and CPU ONNX parity gate. This isolates
domain breadth as the intervention. P1 completed all 2,304 optimizer steps and
produced checkpoint and ONNX bytes, but failed during final report assembly
because the explicitly supplied relative output path could not be normalized
against the absolute repository root. P1 is consumed and cannot rerun. P2 then
reused the exact P1 bytes with zero optimizer steps and recovered the lost
selection report, but failed with 49/96 exact scenes, 21 false positives, 82
misses, three prohibited hits, one marker-artifact hit, artifact precision
`0.8340070217582113`, and recall `0.9549122476437939`. P2 is consumed. A
checksum-bound training-corpus diagnostic reached at most 303/384 exact scenes,
confirming underfitting without public evidence. Final candidate P3 is
preregistered to continue the exact P1 checkpoint for 32 lower-rate epochs with
all data, contracts, loss weights, postprocessing, thresholds, and gates fixed.
P3 execution remains blocked pending a separate authorization. Public
evaluation stays locked. No manifest, model-store entry,
package payload, private validation, production approval, or release eligibility
exists.
