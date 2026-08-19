<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V30 unanimous structure-veto candidate

V30 is a fresh project-owned defect class designed only from the tracked
aggregate V29 public result. V29 retained all 1,792 truths with no misses or
duplicates and kept recognition and every role above threshold, but its one
authorized truth-hidden public run passed only 222/224 scenes because it added
two false regions and two prohibited hits at every fixed threshold.

No V29 case identity, truth row, prediction, tensor content, fixture byte,
private image, Chandler image, or `Generalization` label informed V30. The V29
public archive is consumed and closed. Its bytes cannot be used for V30
training, tuning, diagnosis, or evaluation.

V30 replaces averaged proposal logits with a strict unanimous decision across
three independently initialized project-owned experts: relational attention,
permutation-invariant relation summaries, and a local crop structure veto. The
candidate accepts only the minimum positive-vs-negative margin from the three
routes. A single expert can therefore veto a proposal that the other experts
accept. No V29 checkpoint is reused. Roles remain the source-declared
plot-relative deterministic partition.

P1 preregisters a 1,536-step maximum, an asymmetric false-positive objective,
per-route and worst-route losses, route diversity, and hard-negative emphasis.
Fresh 384-scene training, 192-scene visible-selection, and 256-scene
truth-hidden public families use new seed offsets and disjoint renderer and
degradation identities. The three splits must have zero source-byte overlap.

Selection still requires three consecutive fixed thresholds with every scene
exact, zero false regions, misses, duplicates, and prohibited hits,
recognition exact at least `0.90`, CER at most `0.05`, overall role accuracy at
least `0.90`, every role at least `0.85`, direct stored-byte execution, CPU
execution, tensor hashes, and ONNX parity at most `1e-5`.

The deterministic fixture freezer, checksum-bound archive loader, direct CPU
feature pipeline, single-use P1 training runner, and dynamic ONNX parity check
are implemented. Fresh archives are now frozen from exact source commit
`380d4ece1b48623e60b4d82720d8b421d97349f3`. Their source bundle is
`b830b36be890ed53cb10f43a86961f4412bc238a97bd046d228ead18767cf8fe`,
and the train, selection, and sealed-public source-byte overlap counts are all
zero.

P1 consumed its single authorized CPU training run and visible-selection
evaluation. It passed all 192 scenes exactly at all five fixed thresholds with
zero false regions, misses, duplicates, or prohibited hits. Recognition exact
was `0.9713541666666666`, CER was `0.004869411243913236`, all role accuracies
were `1.0`, and dynamic ONNX parity was `0.000003337860107421875`. The selected
threshold is `0.55`. Only aggregate evidence is tracked in `P1_RESULT.json`;
the checksum-bound checkpoint, ONNX model, and full run report remain ignored
local evidence.

P1 and its single authorized truth-hidden public run are now consumed. The
runner verified every exact payload and source hash before acquiring its
single-use seal, opened the 256-scene archive once, ran the complete detector,
recognizer, relation, and V30 candidate stream on CPU, removed per-scene shape
evidence, and writes only whitelisted aggregate metrics and tensor-stream
hashes. The public run passed 255/256 scenes and retained 2,047/2,048 truth
regions with zero false regions, duplicates, or prohibited hits at each of the
three fixed public thresholds. One false negative caused the exact-scene and
zero-miss requirements to fail. Recognition exact was `0.98095703125`, CER was
`0.0033769523005487546`, overall role accuracy was `0.99951171875`, and the
minimum role accuracy was `0.99609375`.

The archive read count is exactly one. Aggregate report SHA-256 is
`b04db1e244e672ac0c0b0700f8cf05d42b5e54fb78a96ec33b569ff807b93a1d`;
opened and result seal SHA-256 values are
`b77f16e8e848ea571e5b0fc8c7dcc20de517b31e438fbd416459c8e0e682423d`
and `5253429b13e7dbddda42d5a65c686f2e96f741417f5cd4ead8bd335cf1e3c4d8`.
No case identifiers, truth rows, predictions, fixture bytes, or case-level
failure analysis were emitted or used. V30 is closed. Its public bytes cannot
be rerun, reused, or tuned against. A later OCR attempt requires a fresh defect
class and disjoint preregistered splits designed only from this tracked
aggregate result. Marker, private, manifest, model-store, packaging, approval,
and release gates remain closed.
Synthetic fixtures are training and public-test inputs only and can never
become application graph data.
