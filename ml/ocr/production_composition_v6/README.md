# OCR production composition V6

V6 uses fresh synthetic graph scenes and the exact public-passing detector,
official recognizer, numeric specialist, and crop-matched ambiguity classifier.
It retains V5 official-only tick rescue from detector scores `0.90` through
`0.95`. In the lower `0.85` through `0.90` band it requires exact agreement
between the official and numeric recognizers, valid graph-number grammar, and
matching tick geometry. It also resolves visibly separated ambiguity groups
before applying the exact V3 crop and classifier.

The one-use validation gate failed only its frozen invocation-count assertion.
All 120 scenes had exact detection, with 600 true positives, zero false
positives, zero misses, zero duplicates, and zero prohibited hits. Recognition
exact match was `0.99`, CER was `0.001866832607342875`, role accuracy was
`1.0`, numeric exact match was `0.9958333333333333`, word exact match was
`0.9852941176470589`, and ambiguity exact match was `1.0`. Two official-only
and one two-model consensus rescues were correct.

The frozen gate nevertheless required numeric ONNX calls to be at least the
truth count. The numeric preprocessor correctly returns before inference on
crops without encodable glyph components, so direct evidence recorded 512
numeric calls for 600 accepted regions. This is an instrumentation invariant
defect, not permission to rewrite the consumed report. Validation report
SHA-256 is `b50b8fc1f20da8e589a7436e4d8b41143f85f12a45445fc68ee38483175aa12f`.
Public remains unopened. A fresh revision may change only that invariant to
require direct numeric execution greater than zero, with fresh pixels and all
scientific gates unchanged. No model is approved and direct C# composition,
marker exclusion, model-store, packaging, Chandler, and clean-machine evidence
remain mandatory.
