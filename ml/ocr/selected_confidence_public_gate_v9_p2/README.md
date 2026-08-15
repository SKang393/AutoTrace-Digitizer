# OCR selected-confidence P2 eight-role public gate

This gate follows the consumed P2 selection result and uses no P1 or P2
fixture bytes, case-level evidence, private images, Chandler, article data, or
the `Generalization` label. A secret seed that is generated once and never
serialized materializes 160 fresh synthetic scenes with all eight OCR roles.

The single authorized C# CPU execution must pass every scene with exact region
counts, zero false regions, misses, duplicates, or prohibited-structure hits,
direct input and output tensor hashes for all four payloads, recognition exact
match of at least `0.90`, CER at most `0.05`, overall and per-role accuracy of
at least `0.90`, and each numeric, word, and ambiguity family at least `0.90`.

Even a public-gate pass cannot authorize the marker stage, an artifact-mask
provider, manifests, the production model store, packaging, private Chandler
validation, production approval, or release.
