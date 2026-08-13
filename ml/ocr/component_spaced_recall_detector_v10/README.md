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

P2 was separately preregistered as a bounded training repair. It started from
the exact checksum-bound V9 P2 checkpoint and uses only 7,999 proposals from the
already-frozen 240-scene V10 training split. It changes neither the proposal
algorithm nor the selection thresholds, and the validation and public pixels
remained excluded from training. P2 completed 648 optimizer steps, ONNX export,
CPU parity execution, and validation inference, then failed closed before
sealing metrics because the report gate requested a field the evaluator does
not emit. P2 is consumed and cannot rerun.

P3 executed once as a zero-optimizer evidence-path recovery. It
retains the exact P2 checkpoint and ONNX bytes and changes only the invalid
truth-count lookup before executing the frozen validation through CPU ONNX
again. It passed all 80 scenes and 400 truths with zero false regions, misses,
duplicates, or prohibited hits at threshold `0.95`; CPU parity passed at
`5.7220458984375e-06`. The candidate report SHA-256 is
`3755d9d0308fa9420dd3e0cff5c620fed96407ffdc34e4b59e5c651bf50c7455`.
The single checksum-bound public evaluation is authorized but remains unopened.
No candidate is approved, and all downstream gates remain mandatory.
