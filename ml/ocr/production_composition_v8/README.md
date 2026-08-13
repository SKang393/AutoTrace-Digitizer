# OCR production composition V8

V8 is a bounded repair from exposed V7 validation evidence. It preserves the
exact V6/V7 detector, four model payloads, preprocessing, primary threshold,
official-only rescue, general numeric-consensus rescue, spacing behavior, and
scientific gates.

V7 missed one x-axis `0` at detector score `0.8214316368103027`. Both exact
recognizers returned `0`, the numeric confidence was `0.9999194145202637`, and
both assigned x-tick geometry. V8 therefore adds a lower band from `0.82`
inclusive to `0.85` exclusive that can rescue only the exact digit `0` when the
official and numeric recognizers agree and assign the same x-tick or y-tick
role. Any unmatched rescue remains a prohibited-structure failure.

V7 also exposed `!` and `i` as official confusions for `l` within `O o l I`.
V8 routes these two aliases through the already public-passing, checksum-bound
source-group classifier only when every nonspace character is one of `O`, `o`,
`l`, `I`, `!`, or `i`. It does not add a word whitelist or modify model bytes.

V8 uses fresh V6 renderer indices beginning at 40,000 for validation and 50,000
for public. Public remains hidden and cannot run unless exact V8 validation is
committed and passing. Production approval, model-store promotion, marker-stage
evidence, packaging, private Chandler validation, and clean-machine evidence
remain false or absent.

The one-use validation passed all 128 scenes and 640 truths with exact region
counts, zero false regions, zero misses, zero duplicates, and zero prohibited
hits. Recognition exact match was `0.9953125`, CER was
`0.000873871249635887`, role accuracy was `1.0`, numeric exact match was `1.0`,
word exact match was `0.9917355371900827`, and ambiguity exact match was `1.0`.
The lower zero-consensus band was not needed on this fresh split. Direct CPU
execution recorded 128 detector calls, 640 official-recognizer calls, 547
numeric-recognizer calls, and 21 ambiguity-recognizer calls. Validation report
SHA-256 is `032c6badcac9fbb5a093fd10b665df5e91bca1a2b8124588b8184efa15b196a9`.
The report is consumed and cannot rerun. Its exact committed pass authorizes the
single hidden-public evaluation, without approving any production model.

The single hidden-public evaluation also passed. All 160 scenes and 800 truths
had exact region counts, zero false regions, zero misses, zero duplicates, and
zero prohibited hits. Recognition exact match was `0.99375`, CER was
`0.0013979496738117428`, role accuracy was `1.0`, numeric exact match was
`0.996875`, word exact match was `0.9911894273127754`, and ambiguity exact match
was `1.0`. Direct CPU execution recorded 160 detector calls, 801 official calls,
682 numeric calls, and 26 ambiguity calls. Public report SHA-256 is
`43384271fefedab374613141a858367b00d86a14d41b1d0994a66d602f6329b4`.

Both V8 model gates are consumed. Their pass establishes synthetic Python
composition evidence only. Production approval remains false until the exact
behavior executes through the C# application, independent marker-stage
exclusion evidence passes, all exact payloads and notices resolve through the
production model store and package, and private Chandler plus clean-machine
offline workflows pass.
