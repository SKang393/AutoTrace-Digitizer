# OCR selected-confidence acceptance V9 P2

P2 consumes only the tracked aggregate P1 failure: 480/480 truths, four false
regions, no misses or duplicates, and perfect recognition and role metrics.
It does not inspect P1 case detail or reuse P1 fixture bytes, truth, scene IDs,
private images, Chandler, `Generalization`, or article data.

The unchanged P1 composition remains immutable. P2 adds one post-composition
rule: regions accepted by P1's high detector route must have selected-text
confidence of at least `0.75`. P1 rescue routes and all four model payloads are
unchanged. This is candidate two of the frozen three-candidate defect budget.

A fresh 128-scene, five-role visible selection must have exact region counts in
every scene, zero false regions, misses, and duplicates, direct CPU tensor
hashes for all four payloads, and all frozen recognition and role thresholds.
Even a pass cannot authorize the full eight-role public gate, marker evidence,
artifact-mask approval, manifests, model-store promotion, packaging, private
validation, production approval, or release.

The single authorized P2 selection run was consumed at source commit
`e2c0db4810d27074bd34f82b0a6ab45cc14d2625`. It passed the frozen selection
gate on 128 scenes and 640 truth regions with zero false, missed, duplicate, or
prohibited regions. Recognition exact match was `0.9984375`, CER was
`0.00029239766081871346`, and role accuracy was `1.0`. This is selection
evidence only. The six broader gates recorded in `P2_SELECTION_RESULT.json`
remain mandatory and closed.
