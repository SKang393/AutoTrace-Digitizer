# Official PP-OCRv5 image-spacing repair V2

The exact official English recognition-only V1 candidate recognized 190 of 192
fresh selection crops exactly and passed aggregate, CER, role, numeric, word,
and conversion-parity gates. Its two exposed failures preserved the four glyph
identities as `OolI` but omitted three visibly large spaces from `O o l I`.
V1 is consumed and remains immutable.

V2 is a new bounded defect class. P1 retains the exact official ONNX weights,
alphabet, CTC decoding, CPU provider, and all original mandatory thresholds.
It adds one generic postprocessor that may insert whitespace only where the
immutable source crop has a large blank vertical band. The rule receives no
truth string, role, graph position, label whitelist, private image, Chandler
image, or article data. A mandatory regression count rejects any change to a
truth that contains no space.

The selection and truth-hidden public archives use fresh seeds, layouts,
renderer families, and degradation families and are disjoint from the exposed
V1 archives. P1 performs zero optimizer steps and has one selection execution.
Only an exact passing committed selection may authorize the single public gate.
Even a public pass remains unapproved until composed V9 detection, independent
marker-stage safety, C# runtime parity, model-store, packaging, private, and
clean-machine evidence all pass directly.

P1 ran once and failed closed. It preserved every non-space truth, all numeric
labels, and all word labels, but five of 28 ambiguity crops retained a lowercase
`l` where the immutable source pixels showed the top and bottom bars of a
capital `I`. P2 is preregistered to add only that source-shape distinction after
P1 spacing. It does not receive truth, role, graph position, or a label list.

P2 then ran once from committed source and passed all 224 selection crops with
exact match, role, numeric, word, and ambiguity accuracy `1.0`, CER `0.0`, and
zero changes to truths without spaces. The public archive remains unopened and
may be evaluated only once from the separately committed authorization.
