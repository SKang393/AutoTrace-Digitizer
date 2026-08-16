# OCR V21 relational scene proposal and role candidate

V21 is a new defect class after the consumed V9 P3 cross-model selection
failure. It uses only that result's aggregate counts. No predecessor fixture,
truth, case identity, private image, Chandler data, or `Generalization` label is
used.

The isolated architecture change replaces independent proposal decisions with
two learned message-passing blocks over the complete production proposal set.
The model keeps the production component grouping, tight and contextual crop
pixels, original-coordinate geometry, separate proposal and eight-role heads,
and a single dynamic ONNX input.

Training, validation, and sealed-public renderer and degradation families are
disjoint, as are their label vocabularies. A pre-freeze audit covers all 704
source scenes, finds exactly one production proposal for every one of the 5,632
role truths, and retains 28,864 structure negatives. The dynamic proposal axis
exports to ONNX and matches CPU execution for proposal counts 3 and 11 within
the fixed `1e-5` parity gate. All identities must be frozen before the first
optimizer step. Each candidate may execute once. The sealed public set may
execute once only after a selection pass and a separately committed
authorization.

The one-time identities are recorded in `SPLIT_SEAL.json`. Its SHA-256 is
`085c93c73731ca97bc85d4eed52841547e6faab28effa56ca14db90d999b3047`.
The train, selection, and sealed-public archive SHA-256 values are
`c82c527daa4afdadaa477895cd58a93072d73c6ece12b39318f5b6b5f951563d`,
`9d3831f31cdb097f0ec4a2d174ed8d9653d76472394fb5f42c44a17e99990371`,
and `b4ae7547731949ac6df1f9afe3fd83178b3cf9c55c81dbd017592a71d90ddab8`.
The freeze records zero optimizer steps, zero selection evaluations, zero public
evaluations, and no training or public authorization.

`P1_CONFIG.json` and `train_p1.py` fix the one-run P1 training and selection
procedure without authorizing it. The runner rehashes every sealed generator,
dependency, and font source, both non-public archives, its own source bundle,
the candidate configuration, the split seal, and an ancestor source commit
before it can create the consumed-attempt record. It refuses any pre-existing
attempt, checkpoint, ONNX, report, or tracked result. The public archive is not
referenced by the runner. A separately committed checksum-bound authorization
is still required before the first optimizer step.

P1 then executed exactly once from source commit
`a8631be8f0531bcf217a1b9c6f1cc7d0121143de` and consumed all 1,536 fixed
optimizer steps. The 128-scene CPU selection run passed 107 scenes exactly,
retained 1,004 of 1,024 truths, admitted one prohibited false region, missed 20
truths, and produced zero duplicates at threshold `0.45`. Overall role accuracy
was `0.9775390625`; every role passed `0.90`. CPU ONNX parity failed at
`0.000011444091796875`, above the fixed `0.00001` limit. Report SHA-256 is
`f4f5f24ea01148b311c89639e4e76040b8728cb96c667ca1d794e586092d9dc8`;
rejected ONNX SHA-256 is
`ac27ebb25913c2b6da4161d0c3ca34f03110f34a54073d9135076c3718d18c70`.
P1 cannot rerun. The public archive remains unopened with zero evaluations.

P2 is preregistered from P1 aggregate evidence only. It does not inspect P1
case identities, crops, truth records, runtime calls, threshold-by-threshold
case results, or the sealed public archive. The isolated change continues the
exact rejected P1 checkpoint and optimizer state for one additional 384-step
epoch while multiplying only the positive proposal class weight by `2.0`.
Architecture, learning rate, role loss, train and selection archives,
thresholds, exact-count gates, role gates, CPU provider, and `1e-5` ONNX
parity limit stay fixed. P2 has one training execution and one selection
execution. The separate authorization binds source commit `4d3ef2437a84e4cf25fbff7fdcc24c4f263c21f0`,
the exact P1 checkpoint and failed result, every runner source, and both
non-public archives before the first P2 optimizer step. P2 does not authorize
public execution, manifests, model-store promotion, private validation,
production, or release.

P2 then executed exactly once from source commit
`57125818d8fd4d646bf998fa20974b6745765a03` and consumed all 384 additional
optimizer steps. At threshold `0.45`, the 128-scene CPU selection run improved
to 109 exact scenes and 1,006/1,024 retained truths, but it still admitted one
prohibited false region and missed 18 truths. Overall role accuracy was
`0.98046875`; every role remained above `0.90`. CPU ONNX parity also remained
outside the fixed gate at `0.000010251998901367188`. Report SHA-256 is
`c116dbd66dd81d24e3f7db667322c2907ce70eb9781e28771a48b0973f970e50`;
rejected ONNX SHA-256 is
`14b415efffe5e2c6244543811515bef83e19a8bdd596e06f287a810105c87f97`.
The tracked aggregate result SHA-256 is
`8058dd5322244f364567d488c64216a045967548697e2f36d1a5bd62ca2c0833`.
P2 cannot rerun. The public archive remains unopened with zero evaluations.

P3 is the final preregistered candidate. It uses zero optimizer steps and the
exact rejected P2 checkpoint. The isolated calibration multiplies all output
logits by `0.5` and ranks proposals by the geometric mean of proposal-positive
probability and maximum eight-role probability. The scale gives the fixed
`1e-5` parity gate numerical margin while role support can reject a
high-proposal/low-role structure and retain a low-proposal/high-role text
region. Architecture weights, sealed data, thresholds, truth matching, exact
count, role, provider, and parity gates remain fixed. The checksum-bound P3
authorization permits exactly one zero-training export and visible-selection
evaluation. It does not authorize public execution, manifests, model-store
promotion, private validation, production, or release.

Even a public pass cannot approve recognition composition, the marker stage,
an artifact-mask provider, manifests, the model store, packaging, private
Chandler validation, production, or release.
