# Marker-center focal confidence V21

V21 isolates one objective change from the V20 diagnosis: the weighted binary
cross-entropy classification term becomes fixed binary focal loss with standard
RetinaNet values `alpha=0.25` and `gamma=2.0`. Existing positive and
hard-negative sample weights remain outside the focal term. Offset and radius
regression terms remain the V16/V20 formulation.

V20's 21 synthetic training scenes are imported unchanged from
`tail_coverage_v20.training_families`. The V13 proposal manifest and dev split,
3-pixel labels, 0.25 operating threshold, V16 architecture, 2.5-to-8 pixel
radius contract, geometry guard, CPU dynamic ONNX parity counts, Apache-2.0
license, and 0.95 precision/recall bars are frozen. The candidate budget is one;
public, sealed, private, and real-data reads are all zero in this preparation.

The authorized P1 run completed 1,476 optimizer steps. At fixed threshold
`0.25`, precision remains `1.0` and recall improves from V20's
`0.8645833333333334` to `0.9166666666666666`, with zero false positives,
duplicates, or prohibited hits. CPU ONNX parity passes at
`4.76837158203125e-07`, but 8 of 96 dev truths remain missed. V21 is retired
without candidate consumption or a sealed, public, private, article, or
real-data read. The aggregate outcome is `P1_RESULT.json`.
