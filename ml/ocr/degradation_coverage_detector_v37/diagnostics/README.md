# OCR V37 fixed-dev diagnostics

`diagnose.py` holds the authorized V37 ONNX artifact fixed and evaluates the
five-scene V32 synthetic `dev` split. It reports aggregate-only evidence for:

- probability-map pixel segmentation and a fixed threshold sweep;
- connected components under threshold, morphology, and area sweeps;
- source-pixel coverage and overlap prediction disagreement;
- metrics grouped by source image dimensions; and
- the committed aggregate V35 diagnostic ceiling.

The default model path is the ignored V37 P1 ONNX artifact:

```powershell
python -m ml.ocr.degradation_coverage_detector_v37.diagnostics.diagnose `
  --output ml/ocr/degradation_coverage_detector_v37/diagnostics/DIAGNOSTIC.json
```

The command fails closed if the ONNX SHA-256 or CPU execution provider differs
from the authorized run. It reads no private, article, public, or sealed data.
The JSON contains no scene identifiers, pixels, truth rows, or case-level
predictions. The V37 full-box pixel segmentation stage remains the isolated
responsible stage, so another synthetic-only revision is startable.
