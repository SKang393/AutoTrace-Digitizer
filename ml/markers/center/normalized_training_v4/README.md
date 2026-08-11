# Marker-center normalized training V4

Background-invariance V3 applied deterministic median subtraction to the exact
existing marker-center payload. Its one candidate passed CPU ONNX parity but
only 8 of 16 fresh selection scenes were exact. The unchanged weights were not
trained for that normalized input contract.

This is a distinct, synthetic-only training defect class. P1 trains new weights
for the existing radial-topology MLP head on 30 fresh procedural training
scenes after applying the exact V3 patch normalization. Candidate selection uses
16 separate visible validation scenes. The truth-hidden public gate contains 20
new scenes with different family, degradation, seed, and tensor bytes. Chandler,
article images, private images, and previously exposed fixture bytes are absent.

The experiment budget is three candidates. Only P1 is preregistered initially.
P2 or P3 may be preregistered only after a selection failure, using selection
evidence alone. If any candidate opens and fails the single truth-hidden public
gate, the revision ends and no later candidate may be tuned or executed.

Selection and public gates require every scene exact, zero false positives,
zero false negatives, zero duplicate centers, zero prohibited-structure hits,
and CPU ONNX parity no worse than `1e-5`. Threshold selection is limited to the
frozen list in the P1 configuration.

A passing synthetic gate still cannot approve production. Production remains
blocked until the exact preprocessing exists with C# parity, a checksum-bound
approved text and artifact mask provider is composed, the payload passes the
model-store and packaging contracts, Chandler passes once without tuning, and
clean-machine CPU evidence is complete.

## Consumed P1 result

P1 ran once from committed source. It trained on 7,310 examples for 80 epochs
and 4,640 optimizer steps. Threshold `0.6` passed all 16 selection scenes with
128 of 128 markers, and CPU ONNX parity passed at
`1.46031379699707e-06`. The single truth-hidden gate then passed all 20 scenes
with 168 of 168 markers and zero false positives, misses, duplicates, or
prohibited-structure hits.

The ignored ONNX SHA-256 is
`017fca04fa3817596ce3088d73f51003dd3658bc56ec3130e25c92252e6bf739`.
It remains a synthetic-gate candidate only. P1 is consumed and cannot rerun.
P2 and P3 are retired without execution because the public gate is exposed.
No manifest, production-model-store promotion, packaging, or production
approval is permitted from this evidence alone.
