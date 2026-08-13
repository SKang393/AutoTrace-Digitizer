# OCR production composition V6

V6 uses fresh synthetic graph scenes and the exact public-passing detector,
official recognizer, numeric specialist, and crop-matched ambiguity classifier.
It retains V5 official-only tick rescue from detector scores `0.90` through
`0.95`. In the lower `0.85` through `0.90` band it requires exact agreement
between the official and numeric recognizers, valid graph-number grammar, and
matching tick geometry. It also resolves visibly separated ambiguity groups
before applying the exact V3 crop and classifier.

The validation and truth-hidden public gates are single-use. Passing does not
approve a model, create a production manifest, or permit release. Direct C#
composition, marker exclusion, model-store, packaging, Chandler, and
clean-machine evidence remain mandatory.
