<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Direct C# OCR and marker composition gate V2

This preregistered one-time gate is the fresh disjoint successor to the
consumed V1 split. It binds every source PNG and decoded raster, executes the
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

## Preregistered state

No fixture archive has been generated and no model has executed against this
split. The generator uses a fresh non-serialized 128-bit secret, distinct label
sets, distinct renderer and degradation identities, and a distinct scene ID
namespace. V1 fixture bytes, results, and truth are not read or reused.

The runtime clamps only finite activation-boundary drift within the frozen
`1e-5` ONNX parity tolerance and still rejects larger or non-finite violations.
This protocol must be committed before new fixture bytes are sealed. The sealed
identity must then be committed before the one authorized CPU gate execution.
