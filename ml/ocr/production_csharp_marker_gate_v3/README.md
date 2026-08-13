<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Direct C# V11 OCR and marker composition gate V3

This preregistered one-time gate is the disjoint successor to the consumed V2
gate. It binds the exact V11 proposal-role candidate and the unchanged reviewed
recognition and marker payloads before any new sealed fixture is materialized.

The future 48-scene split uses new scene IDs, renderer families, degradation
families, a new unrecorded 128-bit secret, and eight text roles per scene. It
does not read or reuse V1, V2, or V11 public fixture bytes or truth. It contains
only procedural synthetic data and excludes Chandler, Generalization, private
images, and article images.

The C# result must create the text mask passed to the marker stage. A sealed
fixture supplies a checksum-bound procedural structure mask for direct
composition evidence. That fixture mask is not an approved production
artifact-mask provider.

The generator ran once from preregistration commit
`09559668821814af109c5e76b45094607aeb3c0c`. It created 48 scenes with 384
text truths and 392 marker truths. The ignored archive SHA-256 is
`d7fea47e77b46bcce097a1841227b19d4b29cf3cd284f0541f6dc45687386bdc`,
its embedded manifest SHA-256 is
`528b6b9bc107be6dc5ebe3847cc63e83b8158c019e043729d479d16833c3d2ca`,
and the tracked hash-only split seal SHA-256 is
`802b0cae08c71fb622a2d044826c25e4fd25a13079f34f231b54109ce43b86bb`.
The fresh 128-bit secret is not serialized and the seal records zero model
executions. No V1, V2, or V11 public fixture bytes or truth were read or reused.

No V3 model execution is authorized until this hash-only split identity is
committed and pushed. A separate exact authorization commit must then bind the
source commit and fixture hashes before the single gate run.

`PUBLIC_GATE_AUTHORIZATION.json` now authorizes exactly one CPU execution. It
binds sealed-identity commit `804c38468b475856a6c9fd3bf5039c359bd3f147`,
the archive, embedded manifest, tracked seal, all five exact payload hashes,
the exact C# test, and the ignored report path. It does not authorize a rerun,
repair, fixture-mask approval, manifest, model-store promotion, private
validation, production approval, or release eligibility.

The single authorized execution ran from source commit
`458ec387076344229490e58bc98fa0bd07d36af7` and failed closed. The ignored
report SHA-256 is
`a5fc84a4a02c10f39abb156cb33895bb7b6a21c32a98137586b31b9058d7e9a4`.
It records 373/384 OCR true positives, 95 false regions, 11 missed regions,
zero duplicate regions, recognition exact match `0.8880208333333334`, CER
`0.02302158273381295`, role accuracy `0.5260416666666666`, numeric exact match
`1.0`, word exact match `0.8819188191881919`, and ambiguity exact match `1.0`.
The composed marker stage retained 379/392 truths with eight false markers,
13 missed markers, zero duplicates, eight text-derived marker creations, and
zero hits on the separately enumerated prohibited structures. Only 30/48
scenes had exact marker counts. The canonical opened-seal SHA-256 is
`a09a3aa3dece30131d2814aa46262e015f65457956b8fe637bdd3f37b5b5f5d7`;
the fail result-seal SHA-256 is
`ce9c1d30ffbd9126ac9bea9fcbf7f3bfdfc5fe2abd5d77e347e8f358df2c4a1`.
This consumed the authorization. Do not rerun, repair, tune against these
fixtures, manifest, store, package, approve, or privately validate this
composition.

Passing cannot create a model manifest, approve a production model-store
entry, enable normal Auto Detect, package a payload, validate Chandler, or make
the release eligible. All of those gates remain mandatory and fail closed.
