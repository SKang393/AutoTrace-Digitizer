# OCR component ensemble V5

This is a separately preregistered graph-numeric recognition defect class. It
does not rerun or tune the exposed OCR V4 or structural-filter public fixtures.
It freezes a new truth-hidden public split before training and uses only
procedural graph labels, explicit exclusion shapes, and the three reviewed Noto
Sans font payloads already present in the application.

Failure mode: the exhausted structural-filter revision retained exclusion and
role performance but did not transfer decimal-bearing recognition across its
held-out renderer family. Responsible subsystem: per-glyph recognition features
and multi-renderer transfer. Fixed metrics are exact match at least `0.90`, CER
at most `0.05`, role accuracy at least `0.90`, marker exclusion accuracy exactly
`1.0`, and CPU ONNX parity error at most `1e-4`.

P1 combines fixed multiscale pooling, row and column profiles, edge profiles,
radial projections, and six source-geometry features with an MLP. It retains the
fixed pre-classifier structural rejection rule at component-height ratio
`>= 0.75`. The experiment budget is three candidates. P1 ran exactly once from
its committed preregistration, passed the validation-selection gate at
confidence `0.65`, and is consumed. Its validation exact match was
`0.96484375`, CER was `0.04443282801881861`, role accuracy was
`0.9732142857142857`, exclusion accuracy was `1.0`, and CPU ONNX parity error
was `9.298324584960938e-06`. Checkpoint SHA-256 is
`f15f1a282f110b5a32ec112a2fddcc917330806d4e8e8eb4452544e0f69545bd`;
ONNX SHA-256 is
`9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84`.
The sealed public archive was then opened exactly once from that committed
selection state. The CPU run passed exact match `0.994140625`, CER
`0.005759162303664921`, role accuracy `0.9970238095238095`, and exclusion
accuracy `1.0` across 512 positive and 160 exclusion cases. It executed 654
checksum-bound inference calls. Public report SHA-256 is
`a50869bb75f13a39d4601b72e47f49b4b8e684bf4fbb992bd4342e11573e3a20`.
The archive is now exposed and cannot be rerun, repaired, or used for tuning.
Chandler, private article images, external datasets, pretrained weights, and
predecessor public predictions were prohibited from selection and are absent.

No candidate is approved by selection or a public metric alone. Manifest,
model-store, detector composition, marker-creation evidence, provider discovery,
notices, packaging, private validation, and the full release audit remain
mandatory.

## C# runtime boundary

`LocalOnnxComponentTextRecognizer` now implements the V5 per-glyph runtime
shape `[glyph_count,1,24,26] -> [glyph_count,14]`, fixed component isolation,
six source-geometry features, softmax confidence rejection, the `0.75`
structural-height rule, numeric grammar, bounded scheduling, cancellation,
provider evidence, and checksum-bound cache identity. The exact ignored P1
ONNX executed through this recognizer on CPU in `5.9915` ms for a deterministic
two-component probe and returned `11`; the payload SHA-256 remained
`9db95c41ce396e8b2dff3b525556615528a00ca87f4cc531274374b961417c84`.

This is runtime-compatibility evidence only. The source-sealed official-pair
production adapter remains byte-identical to its consumed failed gate, and no
V5 manifest or production-store entry exists. Exact C# preprocessing parity,
a newly preregistered detector composition, independent marker-stage evidence,
private validation, and packaging discovery are still required before the V5
recognizer can be approved or selected by normal Auto Detect.
