# Marker-center train-family coverage V19

V18 P1 completed a train-only hard-positive mining attempt, but its warmup
found zero low-confidence train positives while the unchanged synthetic dev
split still had 29 of 96 misses at confidence `0.25` (`precision=1.0`,
`recall=0.6979166666666666`). The exact V18 aggregate result is bound by
SHA-256 in `protocol.py` and `training/p1.json`.

V19 isolates one change: expand the synthetic **train** stream with four
disjoint families covering the already measured real-range aggregates: small
to very large source dimensions, RGB and RGBA, JPEG compression, the observed
`resize_long` scale range (`0.15x` to `1.50x`), post-resize text heights, and
marker diameters. Examples are generated procedurally from project-owned code.
No image, answer, study, participant, or case identifier from `data/manual
data/` is read.

The generator keeps the bounded V13 patch canvas and applies the measured
resize scale, color/alpha composite, and JPEG round-trip to each synthetic
scene. This preserves the proposal extractor's input contract without
allocating private-corpus-sized canvases; source dimensions remain aggregate
family metadata, never hidden case data.

The V13 proposal extractor, V16 scale-separated CNN, V16 losses, 3-pixel truth
radius, fixed `0.25` confidence threshold, V13 dev split, 2.5-to-8 pixel radius
contract, CPU dynamic ONNX parity checks, Apache-2.0 model license, and 0.95
precision/recall bars remain unchanged. V19 has one candidate budget and zero
sealed or public evaluations at preparation time.

The authorized P1 train/dev run completed 864 optimizer steps. At the selected
fixed threshold `0.25`, precision is `1.0` and recall improves from V18's
`0.6979166666666666` to `0.7916666666666666`, with zero false positives,
duplicates, or prohibited hits. Recall remains below the `0.95` bar. Dynamic
CPU ONNX parity passes at `2.384185791015625e-07`. P1 is retired without a
sealed, public, or private read and without candidate consumption. The tracked
aggregate outcome is `P1_RESULT.json`.
