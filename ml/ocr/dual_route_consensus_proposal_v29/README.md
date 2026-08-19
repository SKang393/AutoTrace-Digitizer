<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V29 dual-route consensus proposal candidate

V29 is a fresh project-owned defect class designed only from the tracked
aggregate terminal V28 public result. V28 passed every visible-selection scene
at all five thresholds, then its one authorized truth-hidden public run passed
188 of 192 scenes at each evaluated threshold. At the selected `0.55`
threshold it retained 1,534 of 1,536 truths, added three false regions, missed
two truths, produced three prohibited-structure hits, and produced no
duplicates. Recognition exact remained `0.978515625`, CER remained
`0.003940553929295204`, and role accuracy remained `0.9986979166666666`.

No V28 case identity, truth row, prediction, tensor content, fixture byte,
private image, Chandler image, or `Generalization` label informed V29. The V28
public archive remains consumed and closed. Its bytes cannot be used for V29
training, tuning, diagnosis, or evaluation.

The V29 proposal model replaces one learned relational route with two
independently initialized routes trained from scratch. The first route retains
pairwise attention over all ordered proposal relations. The second route uses
permutation-invariant mean, maximum, and minimum summaries over the same
truth-independent relation stream. The candidate averages only their proposal
logits. It emits the source-declared plot-relative geometry role partition that
previously produced exact roles, rather than training a role correction against
visible validation.

P1 preregisters an ensemble proposal objective, a separate objective for each
route, a worst-route term, and probability agreement. The complete training
proposal stream is used without a detector prefilter. The route architecture,
loss weights, optimizer, seed, fixed thresholds, direct evidence requirements,
and 1,280-step maximum are frozen before any fixture bytes exist. At most three
candidates may execute under this defect class.

Fresh 320-scene training, 160-scene visible-selection, and 224-scene
truth-hidden public families use new seed offsets, renderer identities,
degradation identities, case identities, and fixture bytes. The three splits
must have zero source-byte overlap. The public archive remains truth hidden and
may be opened at most once only after a candidate passes visible selection and
a separate checksum-bound public runner is committed and authorized.

Selection requires three consecutive fixed thresholds with every scene exact,
zero false regions, misses, duplicates, and prohibited hits, recognition exact
at least `0.90`, CER at most `0.05`, overall role accuracy at least `0.90`,
every role at least `0.85`, direct stored-byte execution, input and output
tensor hashes, CPU execution, deterministic role equality, and ONNX parity at
most `1e-5`.

No V29 fixture archive, split seal, candidate configuration, model payload, or
execution authorization exists at this source checkpoint. Training, visible
selection, public evaluation, marker composition, private validation, manifest
creation, model-store promotion, packaging, production approval, and release
remain unauthorized. Synthetic fixtures are training and public-test inputs
only and are never application graph data.
