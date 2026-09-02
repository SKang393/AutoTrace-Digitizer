# Corrected real-range raw proposal repair V34

The committed V32 raw proposal generator can maximum-match only 61 of 86
corrected real-range dev truths, a raw recall of `0.7093023256`. A classifier
cannot recover a region that the proposal generator never emits. V32 and V33
classifier routes are therefore bound as prior failures, and V34 addresses the
proposal ceiling first.

V34 is the smallest deterministic repair: it unions the committed proposal
generator with proposals from fixed 5th-to-95th percentile contrast, expands
each component by one pixel, re-runs the existing line grouping, and removes
duplicate boxes. It retains the committed V32 five-axis family-disjoint
corrected synthetic train/dev data, canonical maximum-cardinality IoU `0.50`
matching, and fixed `0.95` precision and recall bars. No learned detector is
introduced unless this repair is measured and fails.

The source-bound raw-proposal evaluation reads no private, article, public, or
sealed data. It fails: recall remains `0.7093023255813954`, while deterministic
expansion raises false proposals from 367 to 16,770. The aggregate outcome is
`DEV_DIAGNOSTIC.json`.

This diagnostic is not a model candidate and consumes no candidate budget.
