# Graph text-region detector V1

This revision replaces the rejected official detector only. It retains the
reviewed BGR normalization and fixed DB postprocessing values used by the
production OCR adapter.

The failure mode is direct: the official detector produced valid probability
tensors but reached only 22.92% text detection exact and 48.61% composition
exact on its one frozen diagnostic. The responsible subsystem is detector
region quality, not recognition, probability activation, or provider execution.

P1 is a small strided encoder-decoder trained from random initialization on
procedural generic graph labels and graph-structure exclusions. Chandler,
Generalization, private images, article images, external datasets, pretrained
weights, and the diagnostic fixtures are excluded. Training and validation are
fixed before training, while the distinct sealed public archive remains hidden
until a candidate passes every selection gate.

Selection requires exact region count on every fixture, zero false regions,
zero duplicates, zero exclusion regions, CPU execution, and ONNX parity at most
`1e-4`. The DB probability threshold `0.30`, box threshold `0.60`, unclip ratio
`1.5`, minimum side `3`, and maximum regions `1000` are not tuned. A failed P1
is consumed and any repair requires P2 or P3 preregistration.

P1 is consumed. It completed 1,536 optimizer steps, then failed closed because
one CPU ONNX output reached `1.0000001192092896`. A fixed, non-approval clip
diagnostic still produced zero exact validation fixtures, 409 false regions,
and 78 exclusion false regions. P1 cannot be rerun or promoted.

P2 is preregistered against the same validation and unopened sealed archive.
It retains the P1 network layers, optimizer, epochs, learning rate, and fixed DB
values. The isolated quality repair renders at source scale, applies the exact
production 960/128 resize, and samples deterministic context crops. An explicit
output clip enforces the unchanged strict probability contract. P2 starts from
random initialization and does not reuse P1 weights.

P2 is consumed. It passed CPU parity and the strict probability contract, but
its full-box training target was expanded again by DB postprocessing. No text
fixture reached the fixed IoU gate, and two compact-legend gamma exclusions
produced regions. The median predicted height was 3.14 times truth height.

P3 is consumed. It retained the P2 source renderer, network, optimizer, epochs,
learning rate, explicit clip, validation split, and postprocessing. It changed
supervision to a fixed DB shrink map with ratio `0.40` and placed exclusion
crops over every registered graph-structure family. P3 passed CPU ONNX parity,
the strict probability contract, and all 24 exclusion fixtures, but only 39 of
72 text fixtures were exact. Twenty-nine text fixtures had no matched region
and ten false regions remained.

P1 through P3 and the fixed three-candidate budget are exhausted. The sealed
public archive remains unopened. This revision must not be rerun, tuned,
manifested, placed in the production model store, packaged, or approved.

No result in this folder alone can create a model manifest, promote the local
model store, package weights, approve the combined OCR pair, change release
readiness, or authorize the stable version `1.0.0`.
