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
opened and the public evaluator did not run. P1 cannot rerun. P2 and P3 remain
unregistered and require a separately frozen isolated change before either can
be authorized. Production approval, artifact-mask composition, manifest,
model-store, notice, packaging, clean-machine, and end-to-end workflow evidence
all remain absent.
