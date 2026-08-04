<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Sungwoo Kang -->

# Graph-numeric sequence V2

This bounded experiment preserves horizontal glyph alignment and learns dense
blank or character labels at 32 output positions. It is distinct from the V1
CTC-loss experiment.

It uses only the project-owned 5 by 7 procedural vector glyphs. Generated
corpora, checkpoints, ONNX files, and reports are written below `runs/`, remain
ignored, and are not release assets.

```powershell
python -m pytest ml/ocr/sequence_v2/tests -q
python -m ml.ocr.sequence_v2.train --output ml/ocr/sequence_v2/runs/candidate-a
```

Both permitted Goal 19 runs failed the fixed held-out gates. The experiment is
closed and no model manifest was created. Exact results are recorded in
`models/manifest/ocr/GRAPH_NUMERIC_SEQUENCE_V2_EXPERIMENT_AUDIT.md`.
