# OCR production composition V3

This frozen gate composes the exact public-gate-passing V10 spaced-text
detector, official PP-OCRv5 English recognizer plus spacing-P2 source-pixel
postprocess, and numeric V5 recognizer on fresh procedural graph scenes.

The validation split contains 80 scenes and the truth-hidden public split
contains 112 scenes. Their seeds, layouts, renderer identifiers, degradation
families, and fixture bytes are new. Neither split contains Chandler,
Generalization, private or article images, external data, or predecessor
fixture bytes.

The validation gate may execute once. The public gate remains inaccessible
unless the exact passing validation report is committed with all evaluator
sources. Both gates require the three exact ONNX payloads, direct CPU execution
over checksum-bound fixture bytes, exact detection in every scene, zero false
or duplicate regions, and all preregistered recognition thresholds.

No payload is approved by this gate. Direct C# composition, independent marker
composition, model-store, provider, private Chandler, packaging, notices, and
clean-machine evidence remain mandatory. No manifest or release asset may be
created from a Python-only pass.
