# OCR large-margin proposal calibrator V20

V20 is a fresh fail-closed defect class created from aggregate-only consumed V19
P3 evidence. V19 proved one zero-error operating point exists at threshold
`0.65`, but it failed the unchanged requirement for three consecutive passing
thresholds. No V19 case identity, truth, pixel, fixture byte, Chandler image,
private image, article image, or public archive was inspected or reused.

V20 retains the exact frozen V17 P3 detector, official PP-OCRv5 English mobile
recognizer, 31 generic proposal features, five thresholds, recognition gates,
role gates, direct CPU execution, tensor-stream hashing, and `1e-5` ONNX parity
limit. P1 trains a new Apache-2.0 quadratic-lift MLP from scratch. Its only
defect-class change is a preregistered symmetric large-margin loss targeting
positive and negative logit differences of `1.7346010553881064` and
`-1.7346010553881064`, with fixed loss weight `0.25`.

Fresh stored splits contain 192 training, 128 visible validation, and 192
truth-hidden public scenes. They use disjoint seed, renderer, and degradation
families plus deterministic source rejection sampling that requires exactly one
unchanged production proposal per truth before sealing. The train, validation,
and public archive SHA-256 values are
`cc108f7b138e294073a007b6b7d3d68126cd5aaa17c889b1acd5bcb63eac136f`,
`2e3d96b53189d434aff0c68ce1e1a89c31d248354388d39d089b8a17f913f07e`,
and `e15bfbec72dc27464110192f2524cbbba7e944290bf8a36a64abff053e5b5565`.
The public archive is unopened with zero evaluations.

P1 configuration SHA-256 is
`51de16e69364bb54f2818713ee095333c88484e2597fd0ecb307edbcf6e9e56e`;
runner source bundle SHA-256 is
`f4a91b00f3483e5bc7b2b06fb0d0d17311aef2fc7ba3c4c20cffd2fd32fece69`.
The committed canonical budget ledger now authorizes P1 for one execution. The
runner requires the authorization, candidate configuration, and every bound
source file to be committed before it creates an opened seal or training output.
P1 cannot rerun after that seal is opened. A selected candidate would still
require a separate public-gate authorization, independent marker-stage
composition, manifest, model-store, packaging, clean-machine, private
validation, and release evidence.
