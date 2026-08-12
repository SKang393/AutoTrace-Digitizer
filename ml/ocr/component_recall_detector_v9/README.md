# OCR component-recall detector V9

V9 is a separately preregistered detector defect class triggered by the failed
V8 production-composition validation. V8 missed 113 of 320 text regions while
both recognizers were exact on every detected region. The consumed V8
composition validation and unopened V8 composition public bytes are not reused.

V9 retains the production proposal algorithm, `[N,2,32,140]` encoding, V8 fixed
visual and geometry architecture, balanced unweighted cross entropy, optimizer,
epoch count, and CPU ONNX contract. Its isolated change is a fresh procedural
training, validation, and public family with exactly one y tick, x tick, phase
heading, annotation, and legend label in every scene. New axes, ticks, dividers,
markers, connectors, arrows, brackets, legend frames, and compact shapes remain
hard negatives.

P1 ran once and retained all 400 validation truths, but it accepted 11
prohibited structures across ten of 80 scenes at threshold `0.95`. CPU ONNX
parity passed at `7.62939453125e-06`; report SHA-256 is
`0156d908843e107aa7472276d1b08a3723d744380a0985e517a0f6a29c245062`.
Seven false accepts were one repeated scale-degraded line grouping and four
were compact marker-like shapes. Their scores overlap true text, so threshold
calibration cannot pass.

P2 ran once and solved the detection defect at every frozen threshold: 80/80
validation scenes were exact with 400 true regions and zero false regions,
misses, duplicates, or prohibited hits. It still failed closed because its CPU
ONNX parity error was `1.1444091796875e-05`, above the fixed `1e-05` limit.
Report SHA-256 is
`41c06fc71bd6b709f666d8119031e43788e93aece23f7078a1dc114ae8d624bb`.

Only final P3 is now authorized. It loads the exact checksum-bound P2
checkpoint, multiplies both output logits by `0.5`, performs zero optimizer
steps, and executes all 80 validation scenes through the exported ONNX on CPU.
No weights, data, proposals, thresholds, validation, or public-gate variables
change. The truth-hidden public archive has zero evaluations and cannot open
until P3 passes both detection and the original parity limit.

No Chandler, Generalization, private or article image, external dataset,
pretrained weight, downloaded training sample, or predecessor fixture byte is
used. Passing V9 would remain unapproved until a fresh three-model composition,
marker-stage, private, model-store, packaging, provider, and clean-machine gate
also passes.
