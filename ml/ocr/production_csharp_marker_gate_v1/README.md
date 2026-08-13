<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Direct C# OCR and marker composition gate V1

This preregistered one-time gate closes the direct-execution weakness in the
earlier OCR report. It binds every source PNG and decoded raster, executes the
exact four-payload OCR composition and the exact normalized marker-center ONNX
through the shared CPU runtime, records every input and output tensor hash, and
derives marker metrics from the returned marker-stage objects.

The OCR result itself creates the dense text mask supplied to the marker stage.
The sealed fixture supplies a checksum-bound procedural structure mask for
axes, ticks, dividers, brackets, arrows, legend structures, connecting lines,
and intersections. That fixture mask is test composition evidence only. It is
not an approved production artifact-mask provider.

Passing this gate cannot create a model manifest, approve a production-model
store entry, enable normal Auto Detect, package a payload, validate Chandler,
or make the release eligible. Those gates remain mandatory and fail closed.

The split generator creates a new 40-scene composite renderer and degradation
family, generates a 128-bit secret once, does not serialize it, and writes a
single immutable archive plus a tracked hash-only split seal. No Chandler,
Generalization label, private image, or article image is read or generated.
