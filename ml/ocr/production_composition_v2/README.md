# OCR production composition V2

This preregistered gate composes the exact public-gate-passing V9 detector,
official PP-OCRv5 English recognizer plus spacing-P2 source-pixel postprocess,
and numeric V5 recognizer on fresh procedural graph scenes.

The validation set contains 80 scenes and the truth-hidden public set contains
112 scenes. Both were frozen before any composed model execution. Neither split
contains Chandler, Generalization, private images, article images, external
datasets, or predecessor fixture bytes.

The only validation execution failed closed. It found 399 of 400 truth regions
with zero false regions, duplicates, or prohibited hits. One `O o l I`
annotation was missed. Recognition over the detected regions passed its fixed
family thresholds, but exact detection is mandatory. Report SHA-256 is
`7a20ae70e9c970f2d10dd80f03a41ab363424cf1a33d98327e835727b587bed1`.
The hidden public archive remains unopened and this revision cannot rerun.

No payload is approved. A future detector defect class requires new splits and
must still pass direct C# composition, independent marker-stage, model-store,
private Chandler, packaging provenance, and clean-machine offline gates.
