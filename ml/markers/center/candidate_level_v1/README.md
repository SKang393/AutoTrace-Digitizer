# Candidate-level marker-center revision v1

This research revision replaces the exhausted dense-mask FPN defect class with
a candidate-wise patch model. It does not rerun, tune, or recover P1 through P3
from `marker-center-production-repair-v2`.

The proposal stage scans the existing three production planes at a fixed
four-pixel grid. Only locations with nearby ink are classified. Each
`33 x 33` patch contains ink probability, text mask, and artifact mask. The
compact spatial CNN returns marker probability, sub-grid center offsets, and
radius. Raw text and artifact masks are still sampled at the corrected center,
then deterministic radius-aware non-maximum suppression produces one center per
marker.

The internal candidate tensor contract is new and backward compatible with the
existing public project, vision-result, and model-manifest schemas. The current
production runtime remains `marker-center-runtime-v2`. No application adapter
may select `marker-center-candidate-runtime-v1` until the sealed gate, runtime,
model-store, provider, packaging, and clean-machine gates all pass.

## Scientific controls

- Project-owned procedural images only. No Chandler, private article, or
  downloaded image is used.
- Train and validation renderer families and degradations are disjoint.
- The sealed public split uses a secret ignored seed, disjoint families, and
  disjoint degradations. Its fixture bytes and truth stay ignored.
- The tracked seal binds the archive, private manifest, generator sources, and
  selection manifest before training.
- Training can inspect only train and validation data. It verifies the sealed
  archive checksum but does not open its contents.
- The candidate budget is three. P1 is consumed and cannot rerun. It completed
  60 frozen epochs, then its saved checkpoint produced zero exact validation
  scenes, 134 false positives, and seven duplicates at threshold `0.7`. Its
  ONNX export also failed because adaptive `5 x 5` to `3 x 3` pooling is not
  supported by the pinned exporter. No sealed fixture was opened.
- P2 is consumed and cannot rerun. Its export-safe fixed pool and deterministic
  train-only hard-negative refinement passed CPU ONNX parity, removed all 134
  P1 false positives and all seven duplicates, and retained zero prohibited
  hits. At threshold `0.7`, it still missed one of 63 validation markers and
  therefore did not open the sealed gate.
- P3 consumed the final candidate without optimizer steps or weight changes.
  The exact P2 ONNX at threshold `0.2` produced 9/9 exact validation scenes,
  zero false positives, false negatives, duplicates, and prohibited hits, and
  CPU parity maximum absolute error `1.1920928955078125e-6`.
- P3 opened the single authorized sealed public evaluation exactly once. It
  passed 11/16 scenes exactly, with one false positive, two false negatives,
  zero duplicates, and two prohibited divider hits. The public gate failed.
- All three candidate slots and the sealed-gate budget are exhausted. Do not
  rerun, repair, tune, manifest, store, package, or approve this payload.
- Selection compared only the preregistered thresholds. The sealed public gate
  was opened once for the exact selected ONNX hash.
- Approval requires exact counts in every scene, zero false positives, zero
  false negatives, zero duplicates, zero text/axis/tick/divider/bracket/arrow/
  legend/intersection hits, CPU execution, and ONNX parity at most `1e-5`.

Even a scientific pass would remain fail closed until the production adapter,
approved manifest, checksum-bound model-store discovery, notices, provider
evidence, packaging discovery, and clean-machine evidence pass. P3 did not
reach that boundary because its scientific gate failed.
