<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR structural-veto proposal and role V17

V17 is a new fail-closed defect class derived only from aggregate consumed V16
P3 validation evidence. It does not reuse V16 fixture bytes, scene identities,
truth, or case detail.

The fixed V16 base checkpoint is Apache-2.0 project output. V17 freezes every
base parameter and trains only a separate structural-veto branch on fresh
procedural training families. The branch may lower the positive proposal logit
but may not modify role outputs. Selection still requires every scene exact,
zero false regions, misses, duplicates, and prohibited hits, all role gates,
CPU ONNX parity, and a run of at least three adjacent passing thresholds.

No candidate may execute until fresh train, validation, and truth-hidden public
bytes, hashes, runner sources, candidate configuration, and one-use gate are
committed, then separately authorized. No Chandler, private, article, or
`Generalization` data may be used. Production approval and release eligibility
remain false.

The renderer adapter, frozen-base model, candidate runners, split freezer,
aggregate selection evaluator, and one-use public gate are implemented as
source. The fresh split and its source-bound gate configuration are frozen.

P1 is consumed and failed selection. Its single authorized run completed 1,608
optimizer steps, preserved all 1,728 validation truths, emitted no duplicates,
passed CPU ONNX parity, and preserved the frozen role logits. It retained one
false prohibited region at threshold 0.60 while misses began at threshold 0.64,
so it produced no three-threshold zero-error window. No validation case details
or pixels were used to design P2.

P2 is consumed and failed selection after its single authorized 804-step run.
It shifted the all-truth operating point from threshold 0.60 to 0.64 and passed
CPU ONNX parity, but one prohibited false region remained while misses began at
threshold 0.68. It produced no three-threshold zero-error window. No validation
case details or pixels were used to design P3.

P3 is the final budgeted candidate. It is preregistered but not execution-
authorized. It loads and freezes the exact consumed P2 model, then adds one
context-topology veto branch over the existing tight and context crops, row and
column projections, and frozen geometry. Only this new nonnegative positive-
logit veto may train on the unchanged fresh split. Roles, proposals, selection
thresholds, and the public seal remain unchanged. The public gate remains
unauthorized and unopened, and no production approval or release eligibility
exists.
