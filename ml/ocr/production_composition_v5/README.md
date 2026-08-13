# OCR production composition V5

V5 uses fresh synthetic graph scenes and the exact public-passing detector,
official recognizer, numeric specialist, and crop-matched source-group ambiguity
classifier. A rejected proposal is eligible for rescue only when its detector
score is at least `0.90` and below `0.95`, the official recognizer returns a
valid graph number, and geometry classifies it as an x- or y-axis tick.

The one-use validation gate failed. It produced 559 true positives from 560
truth regions, with zero false positives and zero duplicates. Seven bounded
official-tick rescues were correct, overall recognition exact match was
`0.9892857142857143`, CER was `0.00166333998669328`, and numeric exact match was
`1.0`. One exact `40` proposal remained below the fixed rescue score at
`0.8586363196372986`. Three `O o l I` lines were not split into four source
groups because their first inter-glyph gap was six pixels while the V3 adapter
required seven, leaving ambiguity exact match at `0.8421052631578947`.

Validation report SHA-256 is
`c3894907e9354b841baac5ae9d98997b2f486ff3c87c9d831ead9c827f339d84`.
The validation gate is consumed and the truth-hidden public gate remains
unopened. No model is approved and no production manifest was created. A fresh
revision must preregister the bounded lower-score consensus and an explicit
composition source-group segmentation rule. Direct C# composition, marker
exclusion, model-store, packaging, Chandler, and clean-machine evidence remain
mandatory.
