# Line-context ambiguity classifier V2

This revision replaces the retired tight-normalized V1 representation. It renders each fresh procedural `O`, `o`, `l`, or `I` sample inside the complete Noto Sans font line box, normalizes that shared line box, and then takes a fixed `32x32` crop. This preserves case-relative height and baseline without using truth strings, graph position, private images, or Chandler.

P1 is a new Apache-2.0 project-trained four-class CPU ONNX candidate. Selection and the one-use sealed-public test require at least 0.97 overall and macro accuracy, at least 0.95 for every class, ONNX maximum absolute error at most `1e-5`, and zero ONNX argmax mismatches.

No model manifest, production-store promotion, production approval, or release eligibility is implied by a passing classifier gate. Production composition and marker-creation evidence remain separate mandatory gates.
