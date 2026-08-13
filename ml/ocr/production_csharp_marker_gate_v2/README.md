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

## Sealed fixture identity

The generator ran once from preregistration commit
`04227904ef3e4abd66bf58d272712ebbb879fa51`. It created 40 scenes with 200
text truths and 328 marker truths. The ignored archive SHA-256 is
`f7873bc6fca1bd42c216b94d89d267c843ec2133375bd41f13ab4b887538f301`,
its embedded manifest SHA-256 is
`8167cdfbb7d24cd7d3201ed1eeb7c63bb676029f9f85fb7006043fcdaa9bf607`,
and the tracked hash-only split seal SHA-256 is
`e348d645c85156fa5696de9ea9107031990b20d99832a9172cf9a2cf2daeef57`.
The fresh 128-bit secret is not serialized. V1 fixture bytes, results, and
truth were not read or reused.

The runtime clamps only finite activation-boundary drift within the frozen
`1e-5` ONNX parity tolerance and still rejects larger or non-finite violations.
The sealed identity was committed before the one authorized CPU gate
execution.

## One-time result

The only authorized run executed from commit
`89cda2e789f329a68f906ceafaa8d93008458f07` through
`CPUExecutionProvider` and failed closed. All five exact payloads executed and
the ignored direct report records every input and output tensor hash. Report
SHA-256 is
`9875d0e9f82fb2cae2fecde8a1b38653286ee5a7eff92c5c851b85763525d670`.

OCR produced 180 true-positive regions, 119 false-positive regions, 20 misses,
and zero duplicate regions across 200 truths. No scene was region-exact.
Recognition exact match was `0.845`, character error rate was
`0.01195219123505976`, role accuracy was `0.64`, numeric exact match was
`0.9166666666666666`, word exact match was `0.9469026548672567`, and ambiguity
exact match was `1.0`.

The marker stage produced 310 true positives, 19 false positives, 18 false
negatives, zero duplicates, and 14 of 40 exact scenes across 328 truths. It
also created 18 marker predictions from missed text. All explicitly scored
axis, tick, divider, bracket, arrow, legend, line-intersection, and text hit
counters were zero.

Canonical seal key is
`e7120ab3916e84c0f44addad26fdd790a99682180dd8403c5da0afbad91530e1`.
Opened-seal SHA-256 is
`cef6995b9a6bbd591d5866db98a92fa94f2105ecc1ccbe8408bd1d56664292ef`;
result-seal SHA-256 is
`8e79bf42de3c76ddafb0a5b059b6ef524ad417a6a5a06c74241d93304d3cdc63`.
The result is consumed and the split is exposed. It must not be rerun, repaired,
or tuned against. No manifest, model-store promotion, normal Auto Detect,
packaging, private Chandler validation, production approval, or release
eligibility is authorized by this result.
