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

P1 is preregistered for five epochs and at most 1,280 optimizer steps on 256
fresh synthetic training scenes. Selection contains 128 fresh scenes and the
truth-hidden public set contains 192. Source commit
`41702515c1b13a550f04cbfefe1d393a4e2e13e5` froze all three archives exactly
once. Their SHA-256 values are `1ecb29fb...d85a`, `17a130ec...6d99`, and
`bf9bdde0...bb95`; cross-split source-byte overlap is zero and every truth has
exactly one production proposal. Split-seal SHA-256 is
`0f2acbe8320e3e5b2f2997fe3b93388213bd7c790bee7f026c54f95a51c1b42d`.

The checksum-bound P1 runner and config SHA-256
`94732d1a2ccf839db3637b0b94e870692c0882ad0ac16f83495f5ebccdc2714d`
retain the exact rejected detector, reviewed recognizer, thresholds, and
zero-error gates. Detector, recognizer, evidence, crop, and candidate tensor
streams must all be recorded, and CPU ONNX parity must be at most `1e-5`.

No optimizer step or selection has run, and the public archive remains
unopened. Marker composition, manifest creation, model-store promotion,
private validation, production approval, and release eligibility remain
unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
