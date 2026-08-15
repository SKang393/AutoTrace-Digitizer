# OCR V9 P3 cross-model consensus

This is the final candidate in the recognizer-confirmed selected-confidence
defect class. It uses zero optimizer steps and combines exact V10 proposal and
recognition evidence with independently public-gated V11 role evidence. The
candidate is isolated from ordinary Auto Detect.

Only aggregate P2 failure evidence is used. No P2 public image, truth, scene ID,
case detail, Chandler data, article image, or `Generalization` label enters the
candidate, selection, or public identities. Fresh selection and sealed-public
archives are both frozen before any of the five payloads execute.

Selection and public gates retain exact per-fixture counts, zero false regions,
misses, duplicates, and prohibited hits, overall and per-role accuracy at least
`0.90`, recognition exact match at least `0.90`, CER at most `0.05`, numeric,
word, and ambiguity exact match at least `0.90`, and direct per-call CPU tensor
hashes for all five payloads. Selection executes once. Public execution is
forbidden unless selection passes, and then it also executes once.

Even a public pass cannot authorize marker composition, an artifact-mask
provider, manifests, the model store, packaging, private Chandler validation,
production approval, or release.
