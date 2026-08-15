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

P1 is preregistered but not authorized. Its execution requires a separate commit
that binds the exact candidate configuration and source bundle in the canonical
training ledger. If visible selection does not contain at least three consecutive
zero-error thresholds, the public archive stays unopened. A passing public run
would still require independent marker-stage composition, manifests, model-store
discovery, packaging, clean-machine, and private validation evidence before any
production approval.
