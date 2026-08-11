# Marker-center runtime-consistency V2

The radial-feature revision selected P3 with the
`radial-local-consensus-refinement-v1` postprocessor, but its consumed public
evaluator imported the older line-aware postprocessor. The selection runner
source SHA-256 is
`bc0e42656a610d8f167855cc9caadf44a1c86b795ee030fdb754347e759433c6`;
the mismatched public evaluator source SHA-256 is
`5aa28b68beb76373cd2c3bb88f922825337292fbcc93bc059f4e3deb67dcd0b6`.
The old truth-hidden fixtures are exposed and cannot be rerun or used for
tuning.

This revision makes no model or threshold change. P1 is preregistered to copy
the exact radial-feature P3 checkpoint and ONNX, perform zero optimizer steps,
and use its already selected threshold `0.3` and one-pixel local-consensus
postprocessor for both selection and public evaluation. The source checkpoint
SHA-256 is
`6b670a6f29454d7f63527f57210aa918540a817fca156a71b96872ff09aa2787`;
the source ONNX SHA-256 is
`924c555e2f27955c644143125d7abd3b05859ea9928ab9d1e741e0544fa19e8b`.

Thirty train and twelve visible validation scenes were frozen before payload
execution. The truth-hidden public archive contains 20 new scenes across five
new renderer families and three new degradation families. Its SHA-256 is
`668d8274e3544945d1b6384bdd259cdee81942fd5c9cb36daa25c476574427b7`.
The archive contains only procedural graphs. Chandler, private images, article
images, downloaded data, and prior public fixtures are excluded.

P1 executed once from the committed preregistration and is consumed. It passed
CPU ONNX parity with maximum absolute error `2.384185791015625e-06`, but only
8 of 12 validation scenes were exact. The run retained 95 of 96 markers with
three false positives, one false negative, two duplicate detections, and zero
prohibited-structure hits. Candidate report SHA-256 is
`f53ceba950a5603f3bbbad39ad8718b01784d599e59e59df2320cb6327c4c4a1`.

Because exact selection was mandatory, the truth-hidden public archive was not
opened and the public evaluator did not run. P1 cannot rerun.

A single bounded sweep on the visible validation split then isolated the P1
miss and duplicate behavior without opening public data. The missed marker's
best proposal confidence was `0.25655674934387207`, while all three extra
centers were between `6.1597` and `6.3263` pixels from an accepted center. P2
therefore retains the exact checkpoint and ONNX, performs zero optimizer steps,
keeps the masks, local geometry refinement, matching tolerance, and
radius-relative suppression unchanged, and changes only the confidence
threshold from `0.3` to `0.25` plus the minimum duplicate separation from `5.0`
to `6.5` pixels. That frozen combination produced 12/12 exact visible scenes
and 96/96 markers with zero errors in the one diagnostic sweep. Diagnosis
SHA-256 is
`9163904d698e38ca360bc9b7714636e7cbda54c6ce1202f0d07a991fee86a3d1`.

P2 executed once from that committed preregistration. It passed all 12 visible
selection scenes and 96/96 markers with zero false positives, misses,
duplicates, or prohibited hits. CPU ONNX parity passed at
`2.384185791015625e-06`. P2 then opened the truth-hidden archive exactly once
and failed the public gate: 13/20 scenes were exact, with 164 true positives,
three false positives, four false negatives, zero duplicates, and zero
prohibited-structure hits. Public report SHA-256 is
`9013f187982c6f8e492d6cfbbbd28214f116f21e268ca35ad07526ca014ba5dd`.

The public archive is now exposed and cannot rerun or support tuning. P2 is
consumed, and P3 is retired under this revision rather than being adjusted
against public results. Any future marker-center work requires a distinct
preregistered defect class and a fresh sealed split. Production approval,
artifact-mask composition, manifest, model-store, notice, packaging,
clean-machine, and end-to-end workflow evidence all remain absent.
