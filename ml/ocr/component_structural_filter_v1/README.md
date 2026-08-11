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

P1 may execute once after its protocol, source bindings, evaluator, public gate,
and canonical budget entry are committed. It uses the unchanged OCR V4
validation split for selection. The inherited public fixture archive remains
checksum-bound and unopened until P1 passes every validation gate and a second
committed checkpoint authorizes the exact ONNX and selection-report hashes.

Even a passing public gate cannot approve, manifest, store, package, or release
the model. Direct detector composition, marker-creation exclusion evidence,
provider discovery, notices, model-store validation, packaging, and the full
release audit remain mandatory and fail closed.
