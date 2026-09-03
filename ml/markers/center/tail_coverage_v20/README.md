# Marker-center tail coverage V20

V19's checksum-bound synthetic diagnosis found complete proposal coverage but
19 confidence-below-threshold misses at fixed threshold `0.25`, concentrated
in open-square markers, 11-to-12-pixel radii, and intersection-heavy geometry.
One additional miss was a marker-geometry veto. V20 isolates one repair:
fresh, disjoint train-only variants for those three geometry tails.

The four V19 real-range families remain unchanged in the V20 distribution.
The V13 proposal extractor, V16 scale-separated CNN, V16 losses, 3-pixel
labels, fixed `0.25` threshold, V13 dev split, 2.5-to-8 pixel runtime radius,
CPU dynamic ONNX parity, Apache-2.0 license, and 0.95 precision/recall bars
are frozen. A generator guard keeps added line intersections more than 3
pixels from every truth center, and the runner binds the V19 diagnostic's one
geometry-consensus veto.

V20 reuses V19's broad scenes byte-for-byte and uses fresh procedural seeds
from `2_610_000` for the added tails. The authorized P1 run completed 1,476
optimizer steps. At fixed threshold `0.25`, precision remains `1.0` and recall
improves from V19's `0.7916666666666666` to `0.8645833333333334`, with zero
false positives, duplicates, or prohibited hits. Dynamic CPU ONNX parity passes
at `2.384185791015625e-07`, but 13 of 96 dev truths remain missed. V20 is
retired without candidate consumption or a sealed, public, private, article,
or real-data read. The aggregate outcome is `P1_RESULT.json`.

Known limitation: the diagnosed confidence tail persists on 13 dev truths, and
the synthetic intersection overlay cannot represent every real-image geometry.
Further work must diagnose the fixed synthetic misses before another revision.
