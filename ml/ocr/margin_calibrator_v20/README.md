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
P1 executed exactly once from the committed authorization and is consumed. It
passed CPU ONNX parity at `4.76837158203125e-7` but failed selection at every
threshold. The best aggregate operating point retained 823 of 1,024 truths,
missed 201, and produced zero false regions, duplicates, or prohibited hits.
Recognition exact was `0.798828125`, CER was `0.07375396139441083`, role
accuracy was `0.783203125`, and the minimum role accuracy was `0.2421875` after
unmatched truths were counted. Report SHA-256 is
`a9fc28e963efd0a88cf8168a026778fd50a7dcaff0b7672f848261bb60313d91`;
rejected ONNX SHA-256 is
`802eed6d18f8d032d5ef5b9383cb562631da2ac6f137f9356d6e8481115073f5`.

The aggregate result isolates the fixed detector floor as a recall bottleneck:
the calibrator never received 201 truth proposals and therefore could not
recover them. P1 cannot rerun. P2 is not preregistered or authorized. The public
archive remains unopened with zero evaluations, and marker composition,
manifest, model-store, packaging, clean-machine, private validation, approval,
and release evidence remain blocked.
