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

