# Official PP-OCRv5 recognition-only gate V1

This revision evaluates only the exact converted
`en_PP-OCRv5_mobile_rec` ONNX payload on fresh checksum-bound crop bytes. It
does not rerun the consumed official detector and recognizer pair, reuse either
exposed pair split, or make a production approval decision.

The failure mode is that the pair evaluation stopped at detector inference, so
the recognizer has conversion parity but no independent crop-level accuracy,
role, tensor, or fixture-byte evidence. The responsible subsystem is the
recognition payload and its exact BGR, 48-pixel-height, right-padded CTC
contract. P1 reuses the exact reviewed official model with zero optimizer steps
and one evaluation budget.

Selection and truth-hidden public fixtures use new renderer seeds, crop bytes,
font/degradation combinations, graph numerics, participant names, phase
headers, axis titles, annotations, and common `O/o/l/I` ambiguities. Chandler
and private or article images are prohibited. Passing this gate still cannot
create a manifest, enter the production model store, or enable Auto Detect
until C# preprocessing parity, paired detector composition, marker-stage
safety, packaging discovery, and private validation all pass directly.
