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

Validation and public execution are each single use. Public execution is not
authorized unless the exact V7 validation report is committed and passing. No
model manifest is created and production approval remains false. Direct C#
composition, marker-stage exclusions, model-store discovery, packaging,
private Chandler, and clean-machine offline evidence remain mandatory.
