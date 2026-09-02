# Marker-center scale-stratified V17

V17 targets the concrete V16 failure mode: 29 of 96 V13 dev truths had no
proposal confidence at the fixed 0.25 threshold. Independent synthetic
matching showed that both 3px and 5px proposal labels cover all 96 truths, and
offset and NMS diagnostics lost none. V17 therefore retains V16's 3px label
rule and adds deterministic large-radius-stratified positive oversampling.

Positive examples for markers with radius at least 8 are repeated once. V16's
architecture, losses, thresholds, V13 proposal stream, runtime contract, and
dynamic ONNX parity checks are retained.

V16 P1's aggregate result is bound at
`ml/markers/center/scale_classifier_v16/P1_RESULT.json`. It failed dev with
precision `1.0` and recall `0.6770833333333334`, so this scale-stratified repair
is a separately preregistered synthetic-only candidate. The model license is
Apache-2.0. No private, public, or sealed data is read during preparation.

The authorized P1 run completes 936 optimizer steps and passes dynamic CPU ONNX
parity, but dev precision `1.0` and recall `0.6979166666666666` remain below the
fixed gate. V17 is retired without a private, public, or sealed-data read. The
aggregate outcome is `P1_RESULT.json`.
