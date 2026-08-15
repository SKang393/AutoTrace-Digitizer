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

Even a public pass cannot approve recognition composition, the marker stage,
an artifact-mask provider, manifests, the model store, packaging, private
Chandler validation, production, or release.
