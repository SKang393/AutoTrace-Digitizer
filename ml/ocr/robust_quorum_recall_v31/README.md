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

P2 preregisters only the runner correction. It wraps the unchanged detector
and recognizer sessions as contiguous float32 callables while preserving the
exact V30 weights, frozen fixtures, preprocessing, postprocessing, thresholds,
metrics, and gates. Its source bundle is SHA-256
`630336a225eef62b4fd3aae7ae64d65cc77a0f266acc5bead0dad74b0a4292ae`
and its config is SHA-256
`1b4f968dcbd9d3be16ee8cf05b28ecc7545a166dacbd1410c33d8d9839acfabd`.
P2 is not execution-authorized. Selection still requires three consecutive
fixed thresholds with every scene exact, zero false regions, misses,
duplicates, and prohibited hits, recognition exact at least `0.90`, CER at
most `0.05`, overall role accuracy at least `0.90`, every role at least
`0.85`, direct stored-byte execution, CPU tensor hashes, and ONNX parity at
most `1e-5`. Public execution and marker composition,
private validation, manifest creation, model-store promotion, packaging,
production approval, and release remain closed.

Synthetic fixtures are training and public-test inputs only and can never
become application graph data.
