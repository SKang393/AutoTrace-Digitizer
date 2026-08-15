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

The single authorized execution is consumed and failed closed. It passed direct
CPU runtime evidence but reached only 87/160 exact scenes with 1,206/1,280
truths, one false and prohibited region, 74 misses, zero duplicates, recognition
exact match `0.85625`, CER `0.02385599653003687`, and role accuracy `0.6921875`.
No case-level result is tracked or authorized for tuning. The candidate cannot
rerun, be repaired against this corpus, enter a manifest or model store, reach
private validation, or receive production or release approval.
