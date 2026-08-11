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

P1 has one authorized run after the preregistration commit. It retains 2,880
optimizer steps, the exact production DB thresholds, CPU execution, ONNX parity
at `1e-4`, and the all-fixtures-exact selection gate. No threshold sweep or
public evaluation is authorized. P2 and P3 remain unregistered unless a P1
failure supports one isolated preregistered change.

