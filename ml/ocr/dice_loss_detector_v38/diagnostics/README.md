# OCR V38 fixed-dev diagnostics

`diagnose.py` holds the authorized V38 P1 ONNX and checkpoint fixed and
evaluates only the byte-identified five-scene V32 synthetic `dev` split. It
reports aggregate-only evidence for probability-map segmentation, threshold
sweeps, connected-component postprocessing, tile coverage and overlap
disagreement, source dimensions, and V37 aggregate ceilings.

Run from the repository root:

```powershell
python -m ml.ocr.dice_loss_detector_v38.diagnostics.diagnose `
  --output ml/ocr/dice_loss_detector_v38/diagnostics/DIAGNOSTIC.json
```

The command fails closed if the ignored V38 ONNX/checkpoint/result hashes, the
tracked V37 evidence hashes, CPU provider, or fixed V32 dev fingerprint differ.
It does not read private, article, public, or sealed data. The output contains
no scene identifiers, pixels, truth rows, or case-level predictions.

The diagnostic keeps the V38 P1 model and all postprocessing constants fixed;
it is a diagnosis, not a new candidate or training run. The responsible stage
is expected to remain `full_box_pixel_segmentation` when postprocessing and
tiling cannot clear the Tier 1 bars.
