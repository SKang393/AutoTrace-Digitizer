# Source-group ambiguity classifier V3

This revision trains new Apache-2.0 weights on fresh three-group lines using the exact source-group crop adapter consumed by production composition. It exists because V4 proved that the earlier public-passing line-box model did not transfer to this adapter.

Selection and one-use public thresholds are unchanged: at least 0.97 overall and macro accuracy, at least 0.95 per class, maximum ONNX error `1e-5`, and zero argmax mismatches. Passing does not approve production or release.
