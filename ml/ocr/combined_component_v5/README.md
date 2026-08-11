# Combined detector and component OCR V5

This folder owns the next production OCR composition checkpoint. It does not
contain or approve a production model manifest.

`DETECTOR_DIAGNOSTIC_PROTOCOL.json` freezes one non-approval diagnostic before
inference. The diagnostic uses 72 new public synthetic fixtures at 384 by 192
pixels, the exact audited PP-OCRv5 mobile detector, its exact production BGR and
DB postprocessing contract, and CPU execution. It records raw output extrema,
tensor hashes, text-region matches, false regions, and timing. The diagnostic
may characterize the prior probability-tensor failure and guide a later
preregistration. It may not change a production threshold, create a model
manifest, promote the model store, or satisfy a release gate.

The future combined gate must use a different renderer family, seed, and
fixture bytes. Before any production approval, that gate must execute the exact
detector and selected V5 component recognizer over checksum-bound public and
sealed fixtures, prove C# preprocessing parity, derive marker safety from a real
marker-stage run, and be revalidated by the application approval gate.
