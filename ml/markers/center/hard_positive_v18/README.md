# Marker-center hard-positive mining V18

V17 retained perfect precision but reached only `0.6979166666666666` recall.
The synthetic diagnosis found that 29 of 96 truths were below the fixed
confidence threshold, while proposal coverage, matching radius, offsets, and
NMS were not the cause. V18 therefore performs a fixed warmup on the V17 train
examples, scores only train positives, repeats every positive below `0.25`
three times across all scales and styles, and finishes the remaining epochs.

Dev is not constructed or inspected until the train-only mining and final
training complete. V16's architecture, losses, 3px labels, thresholds, V13
proposal stream, runtime/radius contract, canonical 5px matching, dynamic
ONNX parity checks, Apache-2.0 license, and 0.95 bars are retained.

The authorized P1 run completed after one void report-timer repair. Warmup found
zero train positives below confidence `0.25`, so no examples were mined or
repeated. Dev precision remained `1.0` and recall remained
`0.6979166666666666`, identical to V17. V18 is retired without a private,
public, or sealed-data read. The aggregate outcome is `P1_RESULT.json`.
