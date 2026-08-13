# Conservative official-recognizer spacing V3

This checkpoint isolates the postprocess defect exposed by the first production composition V3 validation.
It uses the exact unchanged `en_PP-OCRv5_mobile_rec` ONNX payload and performs zero optimizer steps.

P1 may insert spaces only from source-image blank bands at least five pixels and 0.40 of the ink height.
It reconstructs spacing from all source groups, including partially spaced raw output, but may never change,
add, delete, or recase a recognized nonspace character. Truth, semantic role, graph position, label lists, private
images, and Chandler pixels are forbidden inputs.

The selection and public crops are new procedural fixtures. The sealed public archive may be opened once only
after the committed selection result and canonical training ledger explicitly authorize P1. Passing this
component gate does not approve a model, promote the production model store, or make the release eligible.
