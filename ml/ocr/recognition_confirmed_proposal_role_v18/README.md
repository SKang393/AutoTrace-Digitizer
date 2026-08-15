<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR recognition-confirmed proposal and role V18

V18 is a fresh fail-closed composition defect class. It does not retrain or
rerun the exhausted V17 detector. It binds the exact V17 P3 detector ONNX at
proposal threshold `0.64` and the exact official PP-OCRv5 English mobile
recognizer ONNX. A proposal is retained only when its collapsed nonblank CTC
path has mean selected-character probability of at least `0.60`.

The architecture was chosen from two aggregate visible-only feasibility checks.
No case identity, fixture pixels, public data, private article data, Chandler,
or `Generalization` data was used. The final aggregate check retained all 1,728
true V17 validation proposals and rejected the sole false proposal at the
fixed confidence threshold. This is feasibility evidence, not selection,
public, marker-composition, or production approval evidence.

Fresh V18 fixture bytes are frozen for 192 visible validation scenes and 256
truth-hidden public scenes. The stored validation archive must execute directly
through both exact CPU ONNX payloads. Selection requires every scene to have
exact regions and roles, zero false regions, misses, duplicates, or prohibited
hits, recognition exact match at least `0.90`, CER at most `0.05`, overall role
accuracy at least `0.90`, and every role at least `0.85`. Tensor input and
output streams for both models are hashed.

P1 is consumed and failed visible selection after its single zero-optimizer
execution. It passed the recognition thresholds with exact match `0.96875`, CER
`0.004607852548718441`, role accuracy `0.998046875`, and minimum per-role
accuracy `0.9895833333333334`. It failed the exact region gates with 189/192
exact scenes, 1,533/1,536 truths retained, one prohibited false region, three
misses, and zero duplicates.

The one-candidate budget is exhausted. No rerun, threshold tuning, or public
access is authorized. The public archive remains unopened with zero evaluations.
Production manifest creation, model-store promotion, packaging, marker-stage
composition, private validation, approval, and release eligibility remain false.
