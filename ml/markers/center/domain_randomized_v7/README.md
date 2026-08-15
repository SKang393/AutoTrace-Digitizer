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
domain breadth as the intervention. Exactly one checksum-bound P1 run is now
authorized. P2 and P3 remain unregistered. Public evaluation stays locked until
a separately committed candidate selection and public evaluator authorization.
No manifest, model-store entry, package payload, private validation, production
approval, or release eligibility exists.
