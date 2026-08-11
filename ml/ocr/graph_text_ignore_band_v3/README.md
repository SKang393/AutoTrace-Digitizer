# Graph text ignore-band detector V3

This is a separate three-candidate detector defect class. Balanced-recall V2
exhausted P1 through P3 without opening its public gate. Its final P3 passed
the probability and CPU parity contracts but produced 82 false regions, 20
exclusion false regions, and 13 text misses across 96 selection fixtures.

V3 freezes new procedural renderer and degradation families before training.
It retains the production BGR input, export-safe skip-connected topology, DB
shrink ratio, and all production DB postprocessing thresholds. Its P1 change
addresses contradictory supervision: visible glyph and antialias pixels between
the shrunken DB core and the full text polygon are ignored instead of labeled as
background. Binary loss uses a frozen 3:1 online hard-negative ratio, and Dice
loss is computed only on valid pixels. Empty-target exclusion patches use a
fixed 4096-pixel hard-negative budget.

The training, validation, and sealed-public families are disjoint from V1 and
V2. The corpus is procedural and generic. It contains no Chandler image,
Generalization label, private or article image, external dataset, pretrained
weight, or downloaded training data.

P1 must pass every frozen selection fixture exactly, with zero false regions,
duplicates, and exclusion hits, plus the probability and CPU ONNX parity gates,
before the single sealed-public run can be authorized. No result in this folder
can create a manifest, populate the production model store, enter a package,
approve OCR, change release readiness, or authorize a release by itself.
