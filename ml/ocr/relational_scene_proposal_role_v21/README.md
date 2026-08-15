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

Even a public pass cannot approve recognition composition, the marker stage,
an artifact-mask provider, manifests, the model store, packaging, private
Chandler validation, production, or release.
