# Marker-center line-aware defect class

This revision follows the exhausted dense and candidate-level revisions. It
does not reuse their weights, exposed public fixtures, family IDs, degradation
IDs, thresholds, or candidate IDs.

P1 changed one defect boundary: candidate proposals and regressed centers must
both agree with the text and artifact masks, while a deterministic radial ink
check rejects line-only structure. A dual-branch CNN learns ink and mask
context separately. P1 was consumed when its ONNX preflight failed before the
optimizer was created. It opened no public fixtures, ran no public evaluation,
and produced no checkpoint.

P2 preserved P1's inputs, outputs,
training scenes, validation scenes, hyperparameters, thresholds, and geometry.
Its only model change replaces unsupported adaptive 8-to-3 pooling with a fixed
4-by-4 stride-two average pool that preserves the 3-by-3 head contract. P2 uses
a new seed because P1 produced no weights. Training and validation use only new
procedural families. P2 passed CPU ONNX parity at `8.270144462585449e-07`
with zero false positives, duplicates, or prohibited hits, but missed four of
63 validation markers. It opened no public fixture and is consumed.

P3 consumed the final candidate slot. A single bounded validation diagnostic
found that the exact P2 payload passes all nine scenes at thresholds `0.03`,
`0.05`, and `0.07`. P3 freezes the most conservative passing value, `0.07`,
and reuses the exact P2 checkpoint and ONNX with zero optimizer steps. It
reproduced all nine exact selection scenes and all 63 markers with zero false
positives, misses, duplicates, or prohibited hits. CPU ONNX parity passed at
`8.270144462585449e-07`.

The single authorized public-gate attempt failed closed before a gate seal was
opened. The frozen public configuration omitted the `expected_*` bindings
required by the canonical gate-seal implementation, so candidate-hash schema
validation stopped the command before the public archive was loaded. No output,
metric, opened seal, or result seal exists. The attempt record is checksum-bound
in `PUBLIC_GATE_ATTEMPT_FAILURE.json`. The one-run budget is consumed and this
revision must not be repaired, rerun, tuned, manifested, stored, packaged, or
approved.

The truth-hidden public archive is generated once before training and can be
opened once only after a candidate passes every selection scene and CPU ONNX
parity.

No result from selection alone authorizes production. A future, separately
authorized defect class would still require a compatible frozen public gate,
an independent artifact-mask gate, production adapter execution, a
checksum-bound manifest and model store, notices, packaging discovery, and
clean-machine proof.
