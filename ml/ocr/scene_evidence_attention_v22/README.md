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

P1 permits five epochs and at most 1,280 optimizer steps on 256 fresh synthetic
training scenes. The fixed selection set has 128 scenes and the truth-hidden
public set has 192 scenes. All split renderer, degradation, seed, and source-byte
identities are disjoint. The three-candidate budget, single public execution,
direct CPU tensor hashing, `1e-5` ONNX parity, three-threshold robustness window,
zero false/missed/duplicate/prohibited regions, recognition, CER, overall role,
and per-role gates remain mandatory.

This checkpoint is source-only. Fixture identity is not frozen, training is not
authorized, public execution is not authorized, and no model, manifest, store
entry, package payload, private validation, production approval, or release
eligibility exists. Synthetic fixtures are training and public-test inputs only
and are never application graph data.

After this source is committed unchanged, freeze the one-time identities with:

```powershell
python -m ml.ocr.scene_evidence_attention_v22.prepare_split freeze
```

