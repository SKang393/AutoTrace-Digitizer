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

## Consumed result

The single authorized evaluation opened canonical seal
`fbcd84e6aa8b74fdce2f7ce191e0c336b2b442b3b716afad9706d73d119a84a5`
from source commit `f2a9fc507d3869227c55f0f7e3b6b0df500ca77b`. The first marker call failed
closed because CPU ONNX activation-boundary drift was rejected by a stricter
runtime boundary than the frozen `1e-5` parity tolerance. No gate report was
created. The opened and result seals have SHA-256 values
`379425b78fb3a7c1ccf6ce264e29aca5b81b60b744fa49b09eb057d37adb4b23`
and `56fc0ee34ad05119012df5886955a52e7a400d7c7cb5227af00d1c03dbc639b6`.
This archive is exposed and must not be rerun or used for tuning.

The runtime now clamps only finite boundary drift within the already frozen
`1e-5` parity tolerance and still rejects larger or non-finite violations. That
repair does not change this failed result or authorize production approval. A
new direct composition attempt requires a new preregistration, new disjoint
fixture bytes, a new source commit, and a new one-run seal.
