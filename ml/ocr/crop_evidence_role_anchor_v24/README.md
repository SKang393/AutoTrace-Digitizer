<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V24 crop-evidence role-anchor candidate

V24 is a fresh project-owned OCR proposal and role defect class. It uses only
the tracked aggregate terminal result of exhausted V23 P3. It does not inspect
or reuse V23 case identities, truth records, fixture pixels, private data,
Chandler, or the `Generalization` label.

V23 demonstrated that the 31-value proposal evidence representation could
retain all 1,024 visible-selection truths with strong recognition and role
metrics, but its original P1 and proposal-head-only P3 retained the same three
false and prohibited regions. V24 therefore changes the representation rather
than repeating another objective adjustment.

The isolated V24 architecture adds the exact two-channel tight and context
proposal crop produced by the existing V21 production encoder. Only the first
128 raster columns are used because the existing 31 evidence values already
contain the encoded geometry. A small shared convolutional encoder fuses those
pixels with the quadratic evidence encoding before the role-anchor set context
and separate proposal and role heads. The model remains dynamic over proposal
count, project-owned, and Apache-2.0.

P1 executed once for five epochs and exactly 1,280 optimizer steps on 256
fresh synthetic training scenes. Selection contains 128 fresh scenes and the
still-unopened truth-hidden public set contains 192. Source commit
`41702515c1b13a550f04cbfefe1d393a4e2e13e5` froze all three archives exactly
once. Their SHA-256 values are `1ecb29fb...d85a`, `17a130ec...6d99`, and
`bf9bdde0...bb95`; cross-split source-byte overlap is zero and every truth has
exactly one production proposal. Split-seal SHA-256 is
`0f2acbe8320e3e5b2f2997fe3b93388213bd7c790bee7f026c54f95a51c1b42d`.

The checksum-bound P1 runner and config SHA-256
`94732d1a2ccf839db3637b0b94e870692c0882ad0ac16f83495f5ebccdc2714d`
retained the exact rejected detector, reviewed recognizer, thresholds, and
zero-error gates. Its direct stored-byte selection produced zero false
positives, zero prohibited hits, and zero duplicates at threshold `0.35`, but
missed 22 of 1,024 truths. Recognition exact was `0.9521484375`, CER was
`0.02102300943552392`, and role accuracy was `0.8818359375`; PhaseHeading was
only `0.421875`. CPU ONNX parity also failed at
`1.7821788787841797e-05`. Report SHA-256 is
`a34d696e81facafe65b77697bf02c98cb03af6a3f19b8a571b55b70ded5d15f8`;
tracked result SHA-256 is
`79f4254287b687674e505961762433c99f309683963650ffed6b982cf71f0715`.

P2 is preregistered from those aggregate metrics only. It replaces the
from-scratch joint model with the exact frozen V23 P3 role-anchor backbone and
trains only a compact crop-conditioned residual on the two proposal logits.
The role outputs must remain byte-exact to the parent. Teacher-positive
preservation, teacher-negative improvement, and scene-separation margins
target the P1 recall regression while retaining its structural precision.
The parent checkpoint is bound to SHA-256
`83d7b47a6fa53ea7b5618acb4b0d4bebb9207594967c4e83ad7c1c62c7cc409d`.
P2 retains the same five thresholds, 1,280-step ceiling, direct tensor-stream
evidence, one visible selection, and `1e-5` CPU parity gate.

P1 is consumed and cannot rerun. P2 has not executed. The public archive
remains unopened. Marker composition, manifest creation, model-store
promotion, private validation, production approval, and release eligibility
remain unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
