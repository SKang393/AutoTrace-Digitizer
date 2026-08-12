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

