<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V23 role-anchor set candidate

V23 is a fresh project-owned OCR proposal and role defect class. It uses only
the tracked aggregate terminal result of exhausted V22 P3. It does not inspect
or reuse V22 case identities, truth records, fixture pixels, private data,
Chandler, or the `Generalization` label.

The isolated architecture change replaces all-proposal self-attention with
eight learned role queries. Each query pools one permutation-invariant anchor
from the complete proposal set. Every proposal then attends those anchors and
receives the scene mean and maximum summaries before separate proposal and role
heads. The exact rejected V17 detector, reviewed official English recognizer,
31 evidence features, role order, thresholds, and gates remain fixed.

P1 permits five epochs and at most 1,280 optimizer steps on 256 fresh synthetic
training scenes. The visible selection has 128 scenes and the truth-hidden
public set has 192 scenes. All renderer, degradation, seed, and source-byte
identities will be frozen before training. The three-candidate budget, single
public execution, direct CPU tensor hashing, `1e-5` ONNX parity, three-threshold
robustness window, zero false/missed/duplicate/prohibited regions, recognition,
CER, overall role, and per-role gates remain mandatory.

No fixture identity is frozen and no candidate is authorized yet. Public
execution, marker composition, manifest creation, model-store promotion,
private validation, production approval, and release eligibility remain
unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
