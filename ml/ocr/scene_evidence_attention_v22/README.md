<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# OCR V22 scene-evidence attention candidate

V22 is a new project-owned detector and recognition composition defect class.
It uses only the aggregate terminal results of the consumed V20 P3 and V21 P3
candidates. It does not inspect or reuse their case identities, truth records,
fixture pixels, private data, Chandler, or the `Generalization` label.

The isolated architecture change trains a new attention calibrator from scratch
over the complete production proposal stream. Each proposal carries the frozen
31-feature detector, role, CTC, geometry, and morphology evidence. Two learned
permutation-equivariant attention blocks allow a proposal to be judged in the
context of every other proposal in its scene. The exact rejected V17 detector
and reviewed official English recognizer remain frozen inputs.

The complete stream means every proposal generated from each sealed scene. The
V17 detector logits remain input evidence, but its rejected `0.56` prefilter is
not applied before V22. P1 must therefore bind the exact sealed positive and
negative proposal counts before either optimization or visible selection.

P1 permits five epochs and at most 1,280 optimizer steps on 256 fresh synthetic
training scenes. The fixed selection set has 128 scenes and the truth-hidden
public set has 192 scenes. All split renderer, degradation, seed, and source-byte
identities are disjoint. The three-candidate budget, single public execution,
direct CPU tensor hashing, `1e-5` ONNX parity, three-threshold robustness window,
zero false/missed/duplicate/prohibited regions, recognition, CER, overall role,
and per-role gates remain mandatory.

The fixture identity is frozen from source commit `1111aeb` with zero
cross-split source-byte overlap. The ignored train, visible-selection, and
truth-hidden public archives are checksum-bound by `SPLIT_SEAL.json`.

P1 is consumed and must not run again. It executed all 16,595 training
proposals and exactly 1,280 optimizer steps. CPU ONNX parity passed at
`6.67572021484375e-06`, but visible selection failed at every fixed threshold.
Thresholds `0.35` and `0.45` retained all 1,024 truths but also retained eight
prohibited false regions. Higher thresholds retained the same false regions
and missed one truth. Recognition exact was `0.978515625`, CER was
`0.0036490296898324765`, overall role accuracy was `0.9931640625`, and the
PhaseHeading class was only `0.9453125`. The public archive remains unopened.

P2 is consumed and must not run again. It was preregistered from only the
aggregate P1 result. Its single isolated change added a scene-extrema
acceptance-margin objective: the highest-scoring training negative was
penalized above probability `0.10`, and the lowest-scoring training truth was
penalized below probability `0.90`. Architecture, sealed fixtures, feature
extraction, role loss, thresholds, selection gates, and the 1,280-step ceiling
remained fixed. P1 was not reused as an initialization.

P2 executed all 16,595 training proposals and exactly 1,280 optimizer steps.
CPU ONNX parity passed at `5.364418029785156e-06`, but visible selection again
failed at every fixed threshold. The best fixed threshold, `0.75`, produced 98
exact scenes out of 128, with 1,023 true positives, three false positives, one
false negative, no duplicates, and three prohibited-structure hits.
Recognition exact was `0.978515625`, CER was `0.0038148946757339524`, overall
role accuracy was `0.9716796875`, and Annotation accuracy fell to `0.828125`.
The public archive remains unopened.

P3 is consumed and must not run again. It used only the aggregate P2 result and
retained the exact P2 checkpoint, architecture, stored fixtures, proposal
stream, scene-extrema objective, thresholds, and gates. Its single isolated
change was the preregistered constrained five-epoch continuation schedule.
Three lower-rate proposal-only epochs kept the role head frozen. Two
role-repair epochs then froze the encoder, attention, and proposal head so role
optimization could not change proposal acceptance. The run completed exactly
1,280 optimizer steps.

CPU ONNX parity passed at `7.450580596923828e-06`. At thresholds `0.55`,
`0.65`, and `0.75`, P3 removed every false proposal, duplicate, and prohibited
structure hit, but retained one missed truth. It therefore passed only 107 of
128 scenes exactly with 1,023 of 1,024 truths. Recognition exact remained
`0.978515625`, CER remained `0.0038148946757339524`, overall role accuracy was
`0.9765625`, and the lowest role accuracy was Annotation at `0.9296875`. The
mandatory zero-error threshold window remained empty.

P1 through P3 are consumed and the defect-class budget is exhausted. The
truth-hidden public archive remains unopened with zero evaluations. Public
execution, marker composition, manifest creation, model-store promotion,
private validation, production approval, and release eligibility remain
unauthorized.

Synthetic fixtures are training and public-test inputs only and are never
application graph data.
