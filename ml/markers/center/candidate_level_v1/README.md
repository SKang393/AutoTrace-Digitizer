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
- Only P2 is now preregistered. It keeps the tensor and public contracts, uses
  an export-safe fixed spatial pool, and adds one deterministic hard-negative
  refinement phase using train scenes only. P3 remains unregistered.
- Selection compares only the preregistered thresholds. The sealed public gate
  opens once for the exact selected ONNX hash.
- Approval requires exact counts in every scene, zero false positives, zero
  false negatives, zero duplicates, zero text/axis/tick/divider/bracket/arrow/
  legend/intersection hits, CPU execution, and ONNX parity at most `1e-5`.

The gate remains fail closed after a scientific pass. A passing model still
needs a production adapter, approved manifest, checksum-bound model-store
discovery, notices, provider evidence, packaging discovery, and clean-machine
evidence.
