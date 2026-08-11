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

No result in this folder alone can create a model manifest, promote the local
model store, package weights, approve the combined OCR pair, change release
readiness, or authorize version `1.0.1`.
