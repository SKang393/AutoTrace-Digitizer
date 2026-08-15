<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Direct C# V8 OCR and marker composition gate V4

This preregistered one-time gate is a disjoint successor to the consumed V3
gate. It binds the repaired V8 OCR composition, the unchanged reviewed
recognition payloads, and the normalized marker-center candidate before any
new sealed fixture identity is materialized.

The future 64-scene split uses new scene IDs, renderer families, degradation
families, marker variants, and one unrecorded 128-bit secret. Each scene covers
the five roles exercised by the V8 public composition: y tick, x tick, phase
heading, annotation, and legend text. This does not prove the full eight-role
production vocabulary. Axis-title, participant, and other-role coverage remain
an explicit blocker even if this gate passes.

The fixtures are procedural synthetic graphs only. They do not read or reuse
predecessor fixture bytes, truth, scene IDs, Chandler, Generalization, private
images, or article images.

The C# OCR result must create the text mask passed to the marker stage. A sealed
fixture supplies a checksum-bound procedural structure mask for direct
composition evidence. That fixture mask is not an approved production
artifact-mask provider.

The generator ran once from preregistration commit
`fff9c96e08672541ea62bdbdb2f08b78fe960609`. It created 64 scenes with 320
text truths and 522 marker truths. The ignored archive SHA-256 is
`b916ab47fa2734ffc657c702572c86da5110eabcd730306062236f9fb6001432`,
its embedded manifest SHA-256 is
`58fe558aedee77495e630e7bc1e8658f70b5ee970b70147bccc302b18d515707`,
and the tracked hash-only split seal SHA-256 is
`3ac34dc2b34a111c2714368c3f7356b7c83f15a1aec5d7db15cb37157ab41b87`.
All 128 image and artifact-mask resources match their embedded checksums. The
fresh secret is not serialized, the seal records zero model executions, and
no predecessor fixture bytes, truth, or scene IDs were reused.

No V4 model execution is authorized until this hash-only identity is committed
and pushed. A separate authorization commit must bind that committed identity,
the archive, embedded manifest, seal, exact test, output path, and five payload
hashes before exactly one CPU execution is eligible to run.

`PUBLIC_GATE_AUTHORIZATION.json` now authorizes exactly one CPU execution. It
binds sealed-identity commit `81ba8f3ca9e35ac89544b63b9cf85ab0c82f8d9f`,
the archive, embedded manifest, tracked seal, exact C# test, ignored report
path, and all five payload hashes. The runner additionally requires the
authorization commit to be the direct child of that sealed identity. The
authorization does not permit a rerun or repair, full-role approval,
artifact-mask approval, a manifest, model-store promotion, private validation,
production approval, or release eligibility.

Passing cannot create a model manifest, approve the production model store,
enable normal Auto Detect, package a payload, validate Chandler, or make the
release eligible. Full role coverage, an approved artifact-mask provider,
model-store and packaging discovery, clean-machine execution, and private
Chandler validation all remain mandatory and fail closed.
