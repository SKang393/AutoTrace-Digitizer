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

No fixture archive or split seal exists at preregistration. No V3 model
execution is authorized until the preregistration is committed, pushed, and a
hash-only split seal is committed from zero model executions. A separate exact
authorization commit must then bind the source commit and fixture hashes before
the single gate run.

Passing cannot create a model manifest, approve a production model-store
entry, enable normal Auto Detect, package a payload, validate Chandler, or make
the release eligible. All of those gates remain mandatory and fail closed.
