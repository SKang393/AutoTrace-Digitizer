# OCR spaced component recall detector V10

Composition V2 missed one generic spaced multi-glyph annotation while retaining
399 of 400 text regions with zero false regions, duplicates, or prohibited
structure hits. V10 is a new defect class with fresh train, selection, and
truth-hidden public scene families.

P1 performed zero optimizer steps and retained the exact V9 P3 ONNX bytes. Its
only validation run selected threshold `0.90`, retained 396/400 truth regions,
and produced zero false regions, duplicates, or prohibited hits. Four misses
remained, so P1 failed and cannot rerun. Report SHA-256 is
`549bb167159ff6b2b82a1d922cc57b7f4d716d1f1c872b6fc49772f85c366602`.
The 112-scene public archive remains unopened.

P2 may be preregistered as a distinct training repair using only the already
frozen V10 training split. P3 remains unregistered. No candidate is approved,
and all downstream gates remain mandatory.
