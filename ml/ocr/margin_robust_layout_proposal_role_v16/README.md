# OCR detection V16: margin-robust layout proposal and role model

This is a fail-closed Apache-2.0 model-repair experiment. It is not a production
model, is not approved, and is not release eligible.

V15 consumed its only hidden public evaluation with 217 of 224 scenes exact,
one false prohibited region, one missed truth, zero duplicates, and role
accuracy above the fixed role gates. V16 uses only those aggregate counts and
the V15 report checksum. It does not expose, reuse, or inspect V15 public case
identities, truth, pixels, or failure details.

The isolated change adds an eight-value plot-relative proposal residual and a
signed proposal-margin loss. Selection requires every region and role to be
exact with no forbidden hits at three or more adjacent fixed thresholds. The
selected cutoff is an interior midpoint of the longest passing run. A single
threshold cannot select a candidate.

The renderer implementation is reviewed procedural V15 code executed with new
V16 seed offsets, scene identifiers, renderer family identifiers, degradation
family identifiers, and split sizes. No predecessor fixture bytes are reused.
The new splits contain 640 training, 192 visible validation, and 256 truth-hidden
public scenes. They contain no Chandler, Generalization, private, article, or
downloaded training data.

Before execution, the generated split bytes, hashes, model runner, candidate
configuration, and single-use public evaluator must be committed. Candidate
P1 then requires a separate canonical-ledger authorization. Passing synthetic
selection or public gates will still not create a manifest, approve a model,
populate the production model store, package a payload, or authorize private
validation or release.
