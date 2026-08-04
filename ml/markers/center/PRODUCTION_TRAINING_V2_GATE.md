# Marker-center production repair v2 candidate 1

The committed current ONNX failed the sealed public gate on 2026-08-04. It had
exact counts in one of three fixtures, zero duplicates, and zero prohibited
structure hits. Candidate `P1` is the first and only currently preregistered
experiment in a new at-most-three-candidate revision.

Candidate `P1` keeps the architecture, procedural selection split, immutable
mask channels, and raw-mask max-gated postprocessing unchanged. Training adds
two selection-only losses at hard-negative pixels identified by the pixelwise
maximum of the text and artifact input masks:

- suppress center probability with weight `1.0`;
- require artifact probability with weight `0.75`.

The consensus threshold is `0.20`. Public data is not used for training,
threshold selection, or candidate selection. The full fixed configuration is
`training/production-repair-v2-p1.json`.

Execution is blocked until the preregistration, canonical budget, and exact
runner source bundle are committed and clean. This is intentional. An
uncommitted training run cannot become scientific evidence.
