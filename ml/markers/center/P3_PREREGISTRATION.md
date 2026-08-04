# Marker-center production repair v2 candidate P3

P2 achieved exact counts, F1 `1.0`, zero false positives, zero false
negatives, zero duplicates, and CPU ONNX parity `3.814697265625e-06` on all
six selection scenes. It failed the hard-negative gate because its
`validation-short_tick_joint_validation-1` annotation placed a
`line_intersection` at `(110,84)`, exactly 8 pixels from the true marker at
`(110,76)`. The correct marker therefore met the inclusive hard-negative hit
rule. P2 is consumed and rejected. The public gate was not opened.

P3 is the final candidate in this defect-class budget and changes one factor:
the six added tick and joint structures move from y coordinates `84..90` to
`108..114`. Their measured minimum distance from every true center is 32
pixels. The short-structure drawing, family counts, data seeds, architecture,
initialization seed, epochs, optimizer, loss weights, mask consensus,
postprocessing, and threshold order remain frozen to P2.

The fixed P3 selection manifest contains 12 training scenes and 6 validation
scenes, includes no public split, uses no private data, and has SHA-256
`5d2192aa2cefcb646abcddea4b87be05898a242d2d8f43a6b23100ec36cbfe02`.

P3 must refuse before output until its dataset, config, runner, budget entry,
and sealed dataset manifest are reviewed, committed, and unchanged. Training
may open the sealed public gate only after all selection and CPU ONNX parity
gates pass. No result from P3 alone grants production or packaging approval.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->
