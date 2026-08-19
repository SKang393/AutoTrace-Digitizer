<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V31 robust quorum recall candidate

V31 is a fresh defect class designed only from the tracked aggregate V30
public result. V30 retained 2,047/2,048 truths, produced no false regions,
duplicates, or prohibited hits, and passed 255/256 scenes at every fixed
threshold. One missed truth closed V30. No V30 case identity, truth row,
prediction, tensor content, fixture byte, private image, Chandler image, or
`Generalization` label informed V31.

The isolated V31 change reuses the exact checksum-bound, project-owned V30
route weights with zero optimizer steps. It replaces strict minimum-margin
unanimity with the median positive-vs-negative margin across attention,
invariant relation-summary, and local-structure routes. This tolerates one
underconfident route while still requiring two independent routes to accept a
proposal. The fixed detector, official recognizer, deterministic roles, CPU
provider, tensor contracts, and mandatory gates are unchanged.

Fresh 384-scene train, 192-scene visible-selection, and 256-scene truth-hidden
public families use new seed offsets and disjoint renderer and degradation
identities. They are frozen by split seal SHA-256
`f6f0778071e7761d6a6065e11c9150e63ef8a1b8ee445976da850b1e0656e113`
from source commit `d3a53bfeeefdce69204d26a4bb9962f2cf659a3b`. Every scene has
exactly one production proposal for each of its eight truths, every within-
split source hash is unique, and all three cross-split source-byte overlap
counts are zero. V30 public bytes and case identities cannot be reused. The
truth-hidden archive may be read once only after a separately committed public
runner and a later explicit authorization.

P1 is consumed. It opened the visible-selection archive once, completed zero
optimizer steps, and failed before scoring because the runner passed raw ONNX
Runtime sessions where the frozen evidence pipeline requires callable tensor
adapters. The tracked aggregate result is SHA-256
`b3087ce9a6c0ab7e71351fb3f2b60bde83f910b89476ffa90c9cc60fcb8f2eed`;
no case detail or pixels were inspected, and the truth-hidden public archive
remains unopened.

P2 changed only the runner boundary. It wrapped the unchanged detector and
recognizer sessions as contiguous float32 callables while preserving the exact
V30 weights, frozen fixtures, preprocessing, postprocessing, thresholds,
metrics, and gates. Its source bundle is SHA-256
`630336a225eef62b4fd3aae7ae64d65cc77a0f266acc5bead0dad74b0a4292ae`
and its separately authorized config is SHA-256
`3ccf3b095db754f2e7105e91abfa5f6b236c3a189d04f98032e172d38087ad04`.
P2 is consumed and failed visible selection. Its single direct stored-byte CPU
run completed zero optimizer steps and passed ONNX parity at
`4.76837158203125e-06`. Threshold `0.75` passed all 192 scenes with all 1,536
truths, zero false regions, misses, duplicates, or prohibited hits,
recognition exact `0.9713541666666666`, CER `0.004862953138815208`, and every
role at `1.0`. Thresholds `0.35`, `0.45`, `0.55`, and `0.65` each retained all
truths but admitted one false prohibited region. Only one threshold passed, so
the mandatory three-consecutive-threshold robustness gate failed. The tracked
aggregate P2 result is SHA-256
`34106e7a018be2964d733162b27292cef5db9bb448eaf3e999accbbd6065c4a3`.
No case detail or pixels were emitted or inspected.

P3 is consumed and V31 is exhausted. Its one authorized invocation ran from
commit `5485a72121a30625fb6b35810392bb56087e597c` after the clean 0.0.22
portable checkpoint. The invocation failed before training authorization was
acquired because the committed preflight accepted historical candidate IDs
`[P2, P3]`, while the canonical acquisition contract requires the current
preregistered list to equal `[P3]`. No output directory or training seal was
created, no optimizer step ran, and the selection archive was not opened. The
tracked aggregate P3 result is SHA-256
`2da0952d6d33ea5f0cd445d0470335d5564b1f91955348f3dea61d137cf8c55a`.
The preflight now mirrors the single-candidate acquisition invariant, but the
consumed P3 invocation cannot be rerun. No case detail or pixels were emitted
or inspected. Public execution, marker composition, private validation,
manifest creation, model-store promotion, packaging, production approval, and
release remain closed.

Synthetic fixtures are training and public-test inputs only and can never
become application graph data.
