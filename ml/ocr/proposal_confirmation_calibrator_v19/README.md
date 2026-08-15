# OCR proposal confirmation calibrator V19

V19 is a fresh, fail-closed research revision. It freezes the exact rejected V17
P3 proposal and role detector and the reviewed official PP-OCRv5 English mobile
recognizer, then trains only a small Apache-2.0 proposal confirmation calibrator.
The calibrator consumes 31 generic detector, role, CTC, geometry, and morphology
features. It does not alter role logits or either frozen payload.

The design used only the aggregate consumed V18 P1 result. No V18 case identity,
truth, pixel, fixture byte, public archive, Chandler image, private image, or
article image was used. V19 has fresh stored train and validation bytes plus a
fresh truth-hidden public archive. All three split families and seeds are
disjoint.

P1, P2, and P3 each executed exactly once from separately committed source and
ledger authorizations. P1 and P2 preserved all 1,024 validation truths at their
selected thresholds and passed recognition, role, and CPU ONNX parity gates,
but each left one prohibited false region and no required three-threshold
zero-error window. P2's only isolated change was to increase negative-class
loss weight from `2.0` to `4.0` based on aggregate P1 counts. At threshold
`0.65`, P2 removed the false region while missing six truths.

P3 used only the aggregate committed P1 and P2 results. It retained P2's data,
payloads, seed, negative-class weight, thresholds, optimizer, and 180-step
budget. Its only change was a deterministic quadratic lift that supplied each
of the 31 frozen features and its square to the same 32-unit hidden layer. P3
passed all single-threshold metrics at `0.65`: 128/128 exact scenes, all 1,024
truths retained, zero false regions, misses, duplicates, or prohibited hits,
recognition exact `0.97265625`, CER `0.004033419763756842`, role accuracy `1.0`,
and CPU ONNX parity `7.152557373046875e-07`. It failed the unchanged robustness
gate because thresholds `0.35` through `0.55` retained one prohibited false
region and threshold `0.75` missed three truths, leaving no required three
consecutive passing thresholds. Report SHA-256 is
`fbbd30ce8f078e62eccdacd5fa178d4128619c214fa162146411f46f38eb6232`;
rejected ONNX SHA-256 is
`98c8dd2e42cb3741b4b4ab4008c77200ac225487781ab193931825afb6362a42`.
All three candidates are consumed, the V19 budget is exhausted, and no rerun or
public evaluation is authorized. The public archive remains unopened with zero
evaluations. No marker-stage composition, manifest, model-store promotion,
packaging, clean-machine, private validation, production approval, or release
eligibility exists.
