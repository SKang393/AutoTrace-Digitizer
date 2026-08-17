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

The aggregate P1 result isolated the fixed detector floor as a recall
bottleneck: the calibrator never received 201 truth proposals and therefore
could not recover them. P1 cannot rerun. P2 then executed exactly once as the
preregistered zero-optimizer, byte-preserving evaluation of the exact P1
checkpoint and ONNX over all 5,384 frozen validation proposals. Its sole change
removed the fixed `0.56` prefilter before the learned calibrator. At its best
threshold `0.45`, P2 retained 1,020/1,024 truths, produced four false
prohibited regions, missed four truths, and passed only 68/128 scenes exactly.
CPU ONNX parity passed at `4.76837158203125e-7`. Candidate report, opened-seal,
and result-seal SHA-256 values are
`678c6e021682e331789a12213650c3f7b3f2292083e8a8338431f96cdcc1892e`,
`fcb0e573364d918fe5c814c69c8a2ff92f4e2140cad6e89caad0cfc84dda0798`,
and `b6a454c22483cf951084c3840fff815b0a262ad1a03a38950b740a3441f3e0c7`.

P3 was specified only from those aggregate P2 metrics and consumed the final
candidate slot. Its complete-stream quadratic multitask calibrator completed
all 1,480 optimizer steps and passed CPU ONNX parity at
`2.384185791015625e-6`. At its best threshold `0.65`, it passed 124/128 scenes
with 1,021 true positives, two false prohibited regions, three misses, and zero
duplicates. Recognition exact was `0.990234375`, CER was
`0.0014405070584845865`, role accuracy was `0.9970703125`, and minimum role
accuracy was `0.984375`. No required three-consecutive-threshold zero-error
window existed. Candidate report, rejected ONNX, checkpoint, opened-seal, and
result-seal SHA-256 values are
`be94ae54be1c85a3170add648eb78dd06af065d42d999af4b5cb989b78ebdeb6`,
`431c7677c5536ac66c60180425c8bfce31fabd71d652c65ab22b0749527c9b5f`,
`edccbb021fb8fa39bcfe5225fb5db04a559b80bc28b97497a14109c18c2a488e`,
`559d2dd7960b5fc443f0771def716eb17bb3dd5b8c877c23599ae8bd59b1da01`,
and `fb1b2ff464dabcd828c7857920e4c29dd6603679fcd7686448bbe692e3b5dc6b`.
P1 through P3 are consumed and the V20 budget is exhausted. The public archive
remains unopened with zero evaluations. No rerun, public gate, marker
composition, manifest, model-store promotion, packaging, clean-machine,
private validation, production approval, or release is authorized.
