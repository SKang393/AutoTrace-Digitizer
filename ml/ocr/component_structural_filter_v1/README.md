# OCR component structural-filter V1

This revision is a separate, one-candidate postprocessing defect class. It does
not reopen or extend the exhausted three-candidate OCR V4 training budget. It
reuses the exact failed P3 ONNX bytes and changes no weights.

The only preregistered change rejects a label crop before classification when
any deterministically isolated component occupies at least 0.75 of the
32-pixel canonical label height. Validation evidence showed that the sole P3
false exclusion was a full-height divider classified as digit `1`, while the
numeric digit `1` components remained below the frozen rule. No threshold sweep
is permitted.

P1 executed exactly once after its protocol, source bindings, evaluator, public
gate, and canonical budget entry were committed. It used the unchanged OCR V4
validation split and passed exact match `0.9609375`, CER
`0.04361370716510903`, role accuracy `0.9739583333333334`, and marker exclusion
accuracy `1.0`. The rule structurally rejected all 14 validation dividers,
changed no weights, and used zero optimizer steps.

The one authorized public evaluation then opened the inherited 384-case archive
and failed. Exact match was `0.87890625`, below the required `0.9`, and CER was
`0.15327695560253699`, above the allowed `0.05`. Role accuracy
`0.9192708333333334` and marker exclusion accuracy `1.0` passed, but they cannot
waive either failed text-quality gate. The direct CPU run made 370 inference
calls and is bound by report SHA-256
`a8cd15bf2a3228209ff24d7db77b458decd9e060bb5c15e2acfc35ab111363b4`.

The candidate and public-gate budgets are exhausted. The exposed archive cannot
be rerun, repaired, tuned against, or reused for selection. This revision must
not be manifested, stored, packaged, approved, or released. A future recognition
defect class requires a separate preregistration and a new unexposed sealed split.
