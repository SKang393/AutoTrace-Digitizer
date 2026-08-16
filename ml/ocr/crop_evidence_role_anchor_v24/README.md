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
truth-hidden public set contains 192. All renderer, degradation, seed, and
source-byte identities must be frozen and disjoint before training. The exact
rejected detector, reviewed recognizer, thresholds, and zero-error gates remain
fixed. Detector, recognizer, evidence, crop, and candidate tensor streams must
all be checksum-bound, and CPU ONNX parity must be at most `1e-5`.

No fixture has been materialized, no optimizer or selection has run, and the
public archive remains unopened. Marker composition, manifest creation,
model-store promotion, private validation, production approval, and release
eligibility remain unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
