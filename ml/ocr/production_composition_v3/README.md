# OCR production composition V3

This frozen gate composes the exact public-gate-passing V10 spaced-text
detector, official PP-OCRv5 English recognizer plus spacing-P2 source-pixel
postprocess, and numeric V5 recognizer on fresh procedural graph scenes.

The validation split contains 80 scenes and the truth-hidden public split
contains 112 scenes. Their seeds, layouts, renderer identifiers, degradation
families, and fixture bytes are new. Neither split contains Chandler,
Generalization, private or article images, external data, or predecessor
fixture bytes.

The validation gate executed once through `CPUExecutionProvider` and failed.
It retained 396/400 truth regions across 76/80 exact scenes with zero false
regions, duplicates, or prohibited hits. Recognition exact match was `0.96`,
CER was `0.00797373358348968`, and role accuracy was `0.9825`, but ambiguity
exact match was only `0.8181818181818182` and spacing-P2 improperly changed
three non-space truths. Report SHA-256 is
`905bb12948ce7bdcdba95f4940e9b1b5f97017da6586c808ff5c43e128049ea9`.
The validation gate is consumed and cannot rerun. The public gate requires an
exact passing validation report, so the truth-hidden public archive remains
unopened and must not execute under this revision.

No payload is approved by this gate. Future detector-recall and conservative
spacing defect classes require new frozen splits rather than tuning against or
rerunning these exposed validation scenes. Direct C# composition, independent
marker composition, model-store, provider, private Chandler, packaging,
notices, and clean-machine evidence remain mandatory.
