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

P2 ran once and passed probability and CPU parity at
`4.708766937255859e-06`, but failed selection at 27/112 exact. It produced
115 false regions, including 15 exclusion false regions, missed text in 76
fixtures, and had only three exact text fixtures. A single broad diagnostic
showed that detections stayed centered but the mandatory 1.5 DB unclip produced
a median 1.94 times truth height and 1.28 times truth width. The ignored band
was unconstrained above the 0.30 production threshold.

P3 was the final preregistered candidate. It added a one-sided squared margin only
inside the existing ignored boundary band. Probabilities at or below 0.25 have
zero margin loss, preserving a low-confidence glyph response while keeping the
band below the fixed 0.30 DB contour threshold. The exact P2 model, whole-frame
tile composition, base loss, optimizer, seed, total 2,880 optimizer steps, DB
thresholds, selection split, and unopened public archive remain unchanged.

P3 ran once and passed the probability contract and CPU ONNX parity at
`1.5497207641601562e-06`. It improved selection to 92/112 exact and removed
every exclusion false positive, but 19 text fixtures had no matched region, two
text false regions remained, and one text fixture produced multiple regions. A
single broad diagnosis found failures across all four held-out structure and
degradation groups. The three-candidate V3 budget is exhausted, and the public
archive remains unopened. No result in this folder can create a manifest,
populate the production model store, enter a package, approve OCR, change
release readiness, or authorize a release by itself.
