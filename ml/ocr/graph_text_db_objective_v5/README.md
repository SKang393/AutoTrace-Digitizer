# Graph text DB-objective detector V5

The stride-4 V4 defect class consumed P1 through P3 without opening its public
gate. Its best candidate, P2, passed the probability and CPU ONNX parity
contracts but left 24 missed text fixtures, 15 false regions, three split text
regions, and one exclusion false region across the fixed 136-fixture selection
set. P3's full-resolution branch regressed those metrics. No V4 candidate may
rerun.

V5 is a new defect class. It tests whether the single shrink-map objective was
unable to learn both connected DB contours and tight geometry. P1 adds a second
threshold-map head and differentiable binary-map supervision. The loss is fixed
before training to shrink, threshold, and binary weights `5:10:1` with `k=50`.
The exported model still returns only one shrink-probability map, so the
production ONNX and DB postprocessing contracts do not change.

The training, validation, and truth-hidden public renderer and degradation
families are new and disjoint from V1 through V4. They contain only procedural
generic labels and graph-like exclusions. They contain no Chandler image,
`Generalization` label, private or article image, external dataset, downloaded
training data, pretrained weight, or public-test truth visible to the runner.

P1 consumed its one authorized run. It passed the probability contract and CPU
ONNX parity at `1.6392e-6`, but failed selection with only 47 of 136 fixtures
exact. It produced 95 false regions, including three exclusion false regions,
and only ten text fixtures were exact. The single checksum-bound visible-split
diagnostic found 83 centered below-IoU text boxes whose post-unclip dimensions
were median 1.824 times truth height and 1.237 times truth width. The public
archive remains unopened.

That diagnosis exposed a bounded omission in P1's isolated change. P1 replaced
V4's shrink objective but did not retain V4's one-sided constraint inside the
ignored shrink-to-source boundary. P2 restored only that squared margin at the
fixed `0.25` ceiling while retaining the dual heads, DB weights, model, frozen
data, optimizer, seed, 2,880 steps, production thresholds, and unopened public
archive. Its one authorized run passed the probability contract and CPU ONNX
parity at `8.3447e-7`. It eliminated all exclusion false regions and improved
exact fixtures from 47 to 84 of 136, but still missed 51 text fixtures,
produced 48 text false regions, and split four text fixtures into multiple
regions. P2 is consumed, unapproved, and may not rerun.

The public archive remains unopened and the public gate remains unauthorized.
One checksum-bound P2 visible-split diagnosis ran with zero threshold sweeps.
It found 44 centered predictions below the IoU gate with median post-unclip
height 1.834 and width 1.186 times truth, versus only seven no-region cases.
All exclusions remained clean. This isolates the remaining defect to uncertain
shrink-map supervision inside the already-defined ignored boundary.

P3 is preregistered as the final candidate. It replaces only P2's squared
ceiling margin with explicit negative binary cross-entropy supervision inside
that same boundary at weight `1.0`. The dual heads, base DB weights, model,
frozen data, optimizer, seed, 2,880 steps, production thresholds, and unopened
public archive remain unchanged. P2 may not rerun, and P3 may run only once
after this preregistration is committed.
