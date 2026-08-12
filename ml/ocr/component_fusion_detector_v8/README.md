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

P1 was consumed without training. Its committed ONNX preflight failed because
the legacy PyTorch exporter cannot represent the adaptive pool from 4 by 16 to
4 by 12. It ran zero optimizer steps and did not open the public archive. P2 is
now the only authorized candidate. It replaces that pool with fixed average
pooling from 4 by 16 to 4 by 8 and updates the following linear width. All data,
proposal, encoding, branch, sampling, optimizer, epoch, seed, threshold, and
public-gate contracts remain fixed.

P2 executed exactly once from its committed preregistration. It passed all 72
validation scenes with 288 of 288 truth regions, zero false regions, misses,
duplicates, or prohibited-structure hits. The selected threshold is `0.95`,
and CPU ONNX parity passed at `5.7220458984375e-06` against the `1e-5` limit.
The training and selection run used 1,216 optimizer steps and did not open the
truth-hidden public archive.

The exact selected ONNX SHA-256 is
`e0254920b26784a87369aa25cc4ec387c6544db30bda4f9542b7ce9a8712e431`.
Its single truth-hidden public gate passed all 88 scenes with 352 of 352 truth
regions and zero false regions, misses, duplicates, or prohibited-structure
hits. The gate executed the exact ONNX on CPU 88 times and is sealed against a
rerun. The public report SHA-256 is
`6f13a622ee444442089104f5897e203beabb1b941981e75a42629d0d01acfdde`.

This is synthetic component-level evidence only. It cannot create a production
manifest, model-store entry, package, or release approval without detector and
V5 recognition composition, independent marker-stage evidence, private
validation, provider discovery, notices, packaging, and clean-machine proof.
