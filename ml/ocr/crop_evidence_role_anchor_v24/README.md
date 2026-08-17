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

P2 executed exactly once from its committed preregistration. It used the exact
frozen V23 P3 role-anchor backbone and trained only the compact crop-conditioned
proposal residual for 1,280 optimizer steps. At threshold `0.35`, direct stored
selection execution retained 1,022/1,024 truths with zero false regions,
duplicates, or prohibited hits. Recognition exact was `0.9697265625`, CER was
`0.005793742757821553`, role accuracy was `0.9931640625`, and every role passed.
The parent role outputs were preserved exactly and CPU ONNX parity passed at
`9.5367431640625e-06`. The two misses and empty three-threshold passing window
still failed selection. Report SHA-256 is
`88d5f7d07e629b9f8fc33121fb180a3f6756f24ff803a74133c512466bdf4f95`;
tracked result SHA-256 is
`c102bf6e2ccc26f401cd23666c81f5d8cdff8c9f2ab530b153dcc50b2f6ce317`.

P3 executed exactly once from its committed preregistration. It used only the
V23 P3 and V24 P2 aggregate terminal metrics plus the already frozen P2
teacher-margin contract. It performed zero optimizer steps, reused the exact P2
weights, and required both V23 parent probability at least `0.35` and
P2-minus-parent crop residual margin at least `-0.25` before emitting a fixed
high-margin proposal decision. It preserved every P2 role output and passed CPU
ONNX parity at `9.5367431640625e-06`, but retained only 1,013/1,024 truths with
zero false regions, duplicates, or prohibited hits. Eleven misses left only
114/128 exact scenes and no three-threshold zero-error window. Report SHA-256 is
`36f44860509583adcb8f762d1fb276e0cc5948b19ad070264ef0205144652556`;
rejected ONNX SHA-256 is
`f97f8c7d85413c1c0d0bf04626e2a94df863b99a9ffca6f9db576c380e6dcacc`;
tracked result SHA-256 is
`42a0da2849c05914543d2a12c84a56d1e3e702f7562adc2dec7fc68865f48766`.

P1 through P3 are consumed and cannot rerun. V24 is exhausted before the
public gate, and its 192-scene public archive remains unopened with zero
evaluations. Marker composition, manifest creation, model-store promotion,
private validation, production approval, and release eligibility remain
unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
