# Balanced-recall graph text detector V2

This is a distinct OCR-detection defect class created after
`graph-text-region-detector-v1` exhausted P1 through P3. The prior final model
passed every graph-structure exclusion but produced no matched region on 29 of
72 text fixtures and left ten false regions. Provider execution and the strict
probability contract passed, so the new responsible subsystem is sparse-positive
feature preservation and recall.

P1 replaces the exhausted tiny strided model and unweighted loss with a
skip-connected encoder-decoder, a fixed positive BCE weight of `8.0`, and a
fixed Dice weight of `2.0`. It retains BGR normalization, the DB shrink ratio
`0.40`, probability threshold `0.30`, box threshold `0.60`, unclip ratio `1.5`,
minimum side `3`, and maximum regions `1000`.

Training, selection, and sealed-public data use new procedural renderer and
degradation families. They do not reuse V1 selection or sealed fixtures and do
not contain Chandler, Generalization, private images, article images, external
datasets, pretrained weights, or downloaded training data.

Selection requires all 96 fixtures exact, zero false regions, zero duplicates,
zero exclusion regions, CPU execution, and ONNX parity at most `1e-4`.

P1 ran exactly once and failed selection. It passed the probability contract
and CPU parity at `3.4570693969726562e-06`, but only 52/96 fixtures were exact.
It produced 38 false regions, including nine on exclusions, and still missed
nine text fixtures. Rail/legend and nested-bracket families each contributed
17 false regions. The sealed public archive remained unopened.

P2 retained the exact P1 architecture, data, thresholds, optimizer, seed, and
28-epoch schedule. Its only change was a fixed weight-`2.0` loss on the highest
loss two percent of background pixels in each sample. It ran exactly once and
failed selection. Probability and parity passed, but only 56/96 fixtures were
exact. It produced 41 false regions, including eight on exclusions, and missed
20 text fixtures. Applying hard-negative mining to text samples penalized real
glyph pixels outside the shrunken DB target. The sealed public archive remained
unopened.

P3 retains the exact P2 model, data, thresholds, optimizer, seed, schedule, and
hard-negative constants. Its only change is to apply the hard-negative term to
empty-target exclusion patches. Text patches continue to use only the P1
weighted BCE and Dice objective. P1 and P2 are consumed and P3 is the only
authorized candidate.

No result in this folder can create a production manifest, populate the model
store, package weights, approve the combined OCR pair, change release readiness,
or authorize version `1.0.1` by itself.
