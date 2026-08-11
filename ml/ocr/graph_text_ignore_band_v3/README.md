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

P1 ran once and passed the probability and CPU ONNX parity contracts, but failed
selection at 8/112 exact. It produced 210 false regions, including 36 exclusion
false regions, missed text in 75 fixtures, and had no exact text fixture. The
focused-crop loss approached zero while the same model also failed a same-family
full-frame diagnostic, binding the failure to omitted graph-wide context.

P2 is preregistered before execution. It changes only training composition:
each of the exact same 640 frozen P1 sources contributes three deterministic
whole-frame tiles. The 1,920-sample composition includes 677 target-bearing
text tiles, 763 negative-context tiles from text sources, and 480 exclusion
tiles. The model, loss, optimizer, seed, total 2,880 optimizer steps, DB
thresholds, selection split, and unopened public archive remain unchanged.

P2 must pass every frozen selection fixture exactly, with zero false regions,
duplicates, and exclusion hits, plus the probability and CPU ONNX parity gates,
before the single sealed-public run can be authorized. No result in this folder
can create a manifest, populate the production model store, enter a package,
approve OCR, change release readiness, or authorize a release by itself.
