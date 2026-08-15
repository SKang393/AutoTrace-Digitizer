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

P1 and P2 each executed exactly once from separately committed source and ledger
authorizations. Both preserved all 1,024 validation truths at the selected
threshold and passed recognition, role, and CPU ONNX parity gates, but each left
one prohibited false region and no required three-threshold zero-error window.
P2's only isolated change was to increase negative-class loss weight from `2.0`
to `4.0` based on aggregate P1 counts. At threshold `0.65`, P2 removed the false
region while missing six truths. P1 and P2 are consumed. P3 is now
preregistered but not execution-authorized. The public archive is unopened with
zero evaluations.

P3 uses only the aggregate committed P1 and P2 results. It retains P2's data,
payloads, seed, negative-class weight, thresholds, optimizer, and 180-step
budget. Its only change is a deterministic quadratic lift that supplies each of
the 31 frozen features and its square to the same 32-unit hidden layer. This
tests whether the aggregate FP/FN tradeoff reflects missing generic nonlinear
separation rather than insufficient class weighting. P3 must receive a later
separate execution authorization. A passing selection and public run would
still require independent marker-stage composition, manifests, model-store
discovery, packaging, clean-machine, and private validation evidence before any
production approval.
