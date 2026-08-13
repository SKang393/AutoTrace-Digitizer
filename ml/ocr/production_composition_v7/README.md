# OCR production composition V7

V7 carries forward the exact V6 detector, recognizers, preprocessing,
postprocessing, thresholds, rescue rules, and source-group ambiguity behavior.
Its only gate change replaces the invalid requirement for one numeric ONNX call
per accepted truth with a direct-execution requirement of at least one call.
The numeric preprocessor is allowed to return before inference when a crop has no
encodable glyph components.

The predecessor V6 validation remains consumed and failed. It passed every
scientific metric across 120 scenes and 600 truths, but its frozen invocation
invariant compared 512 legitimate numeric ONNX calls with all 600 accepted
truths. The V6 public archive remains unopened.

V7 freezes fresh procedural graph scenes from V6 renderer indices that were not
used by V6: validation starts at 20,000 and sealed public starts at 30,000. The
fixture generator binds the transitive V6 renderer sources. The evaluator also
binds the transitive V6 scientific pipeline sources rather than only the thin V7
wrapper.

The one-use validation failed closed. It accepted 619 of 620 truths across 124
fresh scenes with zero false regions, duplicates, or prohibited hits. The sole
detection miss was the x-axis tick `0` in validation scene 79. Recognition exact
match was `0.9903225806451613`, CER was `0.0015028554253080854`, role accuracy
was `0.9983870967741936`, numeric exact match was `1.0`, word exact match was
`0.9914772727272727`, and ambiguity exact match was exactly the required `0.90`.
The repaired runtime invariant passed with 523 direct numeric ONNX calls. Report
SHA-256 is `7d1b2ace57af890fcb95476cc74e1b73f9caf1c22f4c4c9b5a178ab8b80e5dd8`.

The consumed report cannot be rewritten or rerun. Its public archive remains
unopened and is not authorized because validation failed. No model manifest is
created and production approval remains false. A fresh revision may use the
observed missed tick and two ambiguity confusions to preregister a bounded
composition repair, but it must use fresh validation and public fixture bytes.
Direct C# composition, marker-stage exclusions, model-store discovery,
packaging, private Chandler, and clean-machine offline evidence remain
mandatory.
