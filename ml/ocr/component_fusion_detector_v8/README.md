# OCR component-fusion detector V8

V8 is a separately preregistered OCR detection defect class after V7 exhausted
three candidates without opening its public gate. V7 P2 retained every
validation truth region but accepted one compact structure at all tested
thresholds while its training loss approached zero. Only that committed
selection evidence and aggregate failure class were used. No V7 public fixture,
prediction, or pixel was opened by V8 training or selection.

V8 preserves the deterministic V7 proposal and two-channel crop encoding, but
the model no longer treats the twelve repeated scalar geometry columns as image
content. One branch learns from the two 32 by 128 visual crops, a second branch
learns from the twelve scalar values, and their embeddings are fused before the
binary logits. Training uses deterministic balanced batches after a frozen cap
of 24 ordered negative proposals per training scene.

The 256 training, 72 visible validation, and 88 truth-hidden public scenes use
new renderer, degradation, seed, and tensor families. All 1,664 truth regions
have exactly one proposal before training. The scenes add independently varied
compact polygons, open and filled markers, brackets, arrows, legends, axes,
ticks, dividers, connectors, and intersections. Chandler, `Generalization`,
private and article images, external datasets, pretrained weights, downloads,
and predecessor fixture bytes are prohibited.

P1 is the only authorized candidate. It must pass every validation scene with
zero false regions, misses, duplicates, or prohibited-structure hits and CPU
ONNX parity no worse than `1e-5`. Only then may the single truth-hidden public
gate run. A pass remains synthetic evidence only and cannot create a production
manifest, model-store entry, package, or release approval without detector and
V5 recognition composition, independent marker-stage evidence, private
validation, provider discovery, notices, packaging, and clean-machine proof.
